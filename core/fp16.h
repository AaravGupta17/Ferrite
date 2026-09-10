// core/fp16.h
#ifndef FERRITE_FP16_H
#define FERRITE_FP16_H

#include <stdint.h>

/*
 * Half-precision storage formats (IEEE binary16 and bfloat16). Ferrite
 * stores low-precision weights to halve memory footprint; compute always
 * runs in FP32 after the conversion. FP16 uses round-to-nearest-even with
 * full subnormal support (reference bit-remap algorithm); BF16 is the high
 * 16 bits of FP32 with round-to-nearest-even on the discarded mantissa.
 */

/* Convert one float (binary32) to binary16, round-to-nearest-even. */
uint16_t fe_f32_to_fp16(float f);

/* Convert one binary16 back to binary32. */
float fe_fp16_to_f32(uint16_t h);

/* Bulk convert n floats in/out of a half-precision buffer (in place safe). */
void fe_f32_to_fp16_buf(const float *in, uint16_t *out, int n);
void fe_fp16_to_f32_buf(const uint16_t *in, float *out, int n);

/* Convert one float32 to bfloat16 (round-to-nearest-even) and back. */
uint16_t fe_f32_to_bf16(float f);
float fe_bf16_to_f32(uint16_t b);

/* Bulk BF16 conversions (in place safe). */
void fe_f32_to_bf16_buf(const float *in, uint16_t *out, int n);
void fe_bf16_to_f32_buf(const uint16_t *in, float *out, int n);

#endif // FERRITE_FP16_H