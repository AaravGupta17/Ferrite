// core/fp16.c
#include "fp16.h"
#include <string.h>

typedef union { float f; uint32_t u; } F32Bits;

/*
 * Classic bit-level half<->float conversion. Works on the bit pattern only
 * (no FP environment): round-to-nearest-even on the 13 discarded mantissa
 * bits, overflow saturates to infinity, NaN keeps its payload's top bit.
 */

uint16_t fe_f32_to_fp16(float f) {
    F32Bits b;
    b.f = f;
    uint32_t x = b.u;

    uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t exp  = (x >> 23) & 0xFFu;
    uint32_t mant = x & 0x7FFFFFu;

    if (exp == 0xFFu) {                      /* Inf or NaN */
        if (mant == 0) return (uint16_t)(sign | 0x7C00u);
        return (uint16_t)(sign | 0x7C00u | (mant >> 13) | 1u);
    }

    /* Subnormal float32: normalize it until the implicit bit is exposed. */
    if (exp == 0) {
        if (mant == 0) return (uint16_t)sign;          /* +/- 0 */
        uint32_t m = mant;
        int n = 1;
        while (!(m & 0x800000u)) { m <<= 1; n++; }     /* -112 <= n < -110 */
        exp = (uint32_t)(127 - n);
        mant = m;
    }

    int32_t half_exp = (int32_t)exp - 127 + 15;

    if (half_exp <= 0) {                     /* underflow -> subnormal or 0 */
        uint32_t m = mant | 0x800000u;       /* implicit bit */
        if (half_exp < -10) return (uint16_t)sign;
        uint32_t sh = (uint32_t)(1 - half_exp);       /* 1..10 */
        uint32_t half = (m + 0x1000u) >> (sh + 13);   /* RNE on top 10 bits */
        return (uint16_t)(sign | half);
    }

    if (half_exp >= 0x1F) return (uint16_t)(sign | 0x7C00u);  /* overflow */

    /* Normal case with round-to-nearest-even on the 13 LSBs. */
    uint32_t h = ((uint32_t)half_exp << 10) | (mant >> 13);
    uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x0FFFu || (rem == 0x1000u && (h & 1u))) h++;
    if ((h & 0x7C00u) == 0x7C00u) h = sign | 0x7C00u;        /* rounded Inf */
    else h = sign | (h & 0x7FFFu);
    return (uint16_t)h;
}

float fe_fp16_to_f32(uint16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exp  = (uint32_t)(h >> 10) & 0x1Fu;
    uint32_t mant = (uint32_t)h & 0x3FFu;

    uint32_t out;
    if (exp == 0) {                          /* zero or subnormal half */
        if (mant == 0) {
            out = sign;
        } else {
            int e = -1;
            while (!(mant & 0x400u)) { mant <<= 1; e++; }
            uint32_t f_exp = (uint32_t)(127 - 15 - e);
            out = sign | (f_exp << 23) | ((mant & 0x3FFu) << 13);
        }
    } else if (exp == 0x1F) {                /* Inf or NaN */
        out = sign | 0x7F800000u;
        if (mant) out |= (mant << 13) | 0x400000u;
    } else {
        out = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    F32Bits b;
    b.u = out;
    return b.f;
}

void fe_f32_to_fp16_buf(const float *in, uint16_t *out, int n) {
    for (int i = 0; i < n; i++) out[i] = fe_f32_to_fp16(in[i]);
}

void fe_fp16_to_f32_buf(const uint16_t *in, float *out, int n) {
    for (int i = 0; i < n; i++) out[i] = fe_fp16_to_f32(in[i]);
}

uint16_t fe_f32_to_bf16(float f) {
    F32Bits b;
    b.f = f;
    uint32_t x = b.u;
    /* Round-to-nearest-even on the 16 low bits, then truncate. */
    uint32_t lsb = (x >> 16) & 1u;
    uint32_t rnd = 0x7FFFu + lsb;
    uint32_t y = (x + rnd) >> 16;
    /* NaN keeps its payload's high bit; overflow stays Inf. */
    return (uint16_t)y;
}

float fe_bf16_to_f32(uint16_t b) {
    F32Bits v;
    v.u = (uint32_t)b << 16;
    return v.f;
}

void fe_f32_to_bf16_buf(const float *in, uint16_t *out, int n) {
    for (int i = 0; i < n; i++) out[i] = fe_f32_to_bf16(in[i]);
}

void fe_bf16_to_f32_buf(const uint16_t *in, float *out, int n) {
    for (int i = 0; i < n; i++) out[i] = fe_bf16_to_f32(in[i]);
}