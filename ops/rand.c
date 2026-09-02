/* ops/rand.c */
#include "ops.h"
#include <math.h>

/*
 * Deterministic pseudo-random streams, per the Stage 2 math-backend
 * contract: a fixed seed must reproduce identical output across runs.
 * We use a splitmix32-style generator (no external RNG state, fully
 * determined by the seed), so results are reproducible and portable.
 */

static uint32_t next_u32(uint32_t *state) {
    uint32_t z = (*state += 0x9E3779B9u);
    z = (z ^ (z >> 16)) * 0x21F0AAADu;
    z = (z ^ (z >> 15)) * 0x735A2D97u;
    z ^= (z >> 15);
    return z;
}

/* Uniform in [0,1) — the low 24 bits give 2^-24 granularity. */
static float next_unit(uint32_t *state) {
    uint32_t u = next_u32(state) >> 8;       /* keep 24 bits */
    return (float)u * (1.0f / 16777216.0f);
}

FeStatus fe_rand_uniform(FeTensor *out, float low, float high, uint32_t seed) {
    if (!out || !out->data) return FE_ERR_NULL;
    if (out->dtype != DTYPE_FLOAT32) return FE_ERR_DTYPE;
    if (!(high > low)) return FE_ERR_SHAPE;

    uint32_t state = seed;
    int n = fe_tensor_numel(out);
    float *p = (float *)out->data;
    float range = high - low;
    for (int i = 0; i < n; i++) {
        p[i] = low + next_unit(&state) * range;
    }
    return FE_OK;
}

/* Box-Muller transform for a standard normal, then scale/shift. */
FeStatus fe_rand_normal(FeTensor *out, float mean, float stddev, uint32_t seed) {
    if (!out || !out->data) return FE_ERR_NULL;
    if (out->dtype != DTYPE_FLOAT32) return FE_ERR_DTYPE;
    if (!(stddev >= 0.0f)) return FE_ERR_SHAPE;

    uint32_t state = seed;
    int n = fe_tensor_numel(out);
    float *p = (float *)out->data;

    for (int i = 0; i < n; i += 2) {
        float u1 = next_unit(&state);
        float u2 = next_unit(&state);
        /* Guard against the log(0) extreme: u1 in [0,1); clamp to >0. */
        if (u1 <= 0.0f) u1 = 2.220446049250313e-16f;   /* FLT_MIN-ish */
        float mag = stddev * sqrtf(-2.0f * logf(u1));
        float z0  = mag * cosf(2.0f * (float)M_PI * u2) + mean;
        float z1  = mag * sinf(2.0f * (float)M_PI * u2) + mean;

        p[i] = z0;
        if (i + 1 < n) p[i + 1] = z1;
    }
    return FE_OK;
}
