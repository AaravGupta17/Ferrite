/* ops/norm.c — normalization layers (BatchNorm, LayerNorm, GroupNorm). */
#include "ops.h"
#include <math.h>

/*
 * Shared skeleton: the three norms differ only in which axes each
 * normalization unit reduces over. We find the reduction by walking the
 * input with a helper that normalizes one contiguous extent (the set of
 * elements grouped together), applying (x - mean)/sqrt(var+eps)*gamma+beta.
 *
 * For clarity each op lays out its own loops; the common bit is the
 * per-unit statistics and the affine transform.
 */

/* Normalize `count` contiguous values at `src`; gamma/beta are the per-unit
 * affine params (may be NULL to skip scaling/shifting). */
static void normalize_unit(const float *src, float *dst, int count,
                           const float *gamma, const float *beta,
                           float eps) {
    double sum = 0.0, sumsq = 0.0;
    for (int i = 0; i < count; i++) {
        double v = src[i];
        sum += v; sumsq += v * v;
    }
    double mean = sum / count;
    double var  = sumsq / count - mean * mean;
    if (var < 0.0) var = 0.0;
    float inv_std = 1.0f / (float)sqrt(var + eps);
    for (int i = 0; i < count; i++) {
        float x = (src[i] - (float)mean) * inv_std;
        dst[i] = (gamma ? gamma[i] : 1.0f) * x + (beta ? beta[i] : 0.0f);
    }
}

/* BatchNorm: [N, C, H, W]; per-channel params. Channel is stride 1 of the
 * trailing spatial dims, so a unit is one (H*W) plane for one channel. */
FeStatus fe_batchnorm(const FeTensor *in, const FeTensor *gamma,
                      const FeTensor *beta, const FeTensor *mean,
                      const FeTensor *var, FeTensor *out, float eps) {
    if (!in || !out || !gamma || !beta || !mean || !var) return FE_ERR_NULL;
    if (in->ndim < 2) return FE_ERR_SHAPE;
    if (in->dtype != DTYPE_FLOAT32 || out->dtype != DTYPE_FLOAT32)
        return FE_ERR_DTYPE;
    int C = in->shape[1];
    if (gamma->shape[0] != C || beta->shape[0] != C ||
        mean->shape[0] != C || var->shape[0] != C)
        return FE_ERR_SHAPE;

    int spatial = 1;
    for (int d = 2; d < in->ndim; d++) spatial *= in->shape[d];
    int N = in->shape[0];

    const float *p = (const float *)in->data;
    float *o = (float *)out->data;
    const float *g = (const float *)gamma->data;
    const float *b = (const float *)beta->data;
    const float *m = (const float *)mean->data;
    const float *v = (const float *)var->data;

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            const float *src = p + (n * C + c) * spatial;
            float *dst = o + (n * C + c) * spatial;
            float m_c = m[c], inv_std = 1.0f / (float)sqrt(v[c] + eps);
            for (int i = 0; i < spatial; i++)
                dst[i] = g[c] * ((src[i] - m_c) * inv_std) + b[c];
        }
    }
    return FE_OK;
}

/* LayerNorm: normalize over the trailing `norm_ndim` axes.
 * gamma/beta shape must match the trailing norm_ndim axes of `in`. */
FeStatus fe_layernorm(const FeTensor *in, const FeTensor *gamma,
                      const FeTensor *beta, FeTensor *out, float eps) {
    if (!in || !out || !gamma || !beta) return FE_ERR_NULL;
    if (in->dtype != DTYPE_FLOAT32 || out->dtype != DTYPE_FLOAT32)
        return FE_ERR_DTYPE;

    int norm_ndim = gamma->ndim;
    if (norm_ndim < 1 || norm_ndim > in->ndim) return FE_ERR_SHAPE;

    int outer = 1;
    for (int d = 0; d < in->ndim - norm_ndim; d++) outer *= in->shape[d];

    int inner = 1;
    for (int d = in->ndim - norm_ndim; d < in->ndim; d++) inner *= in->shape[d];

    const float *p = (const float *)in->data;
    float *o = (float *)out->data;
    const float *g = (const float *)gamma->data;
    const float *b = (const float *)beta->data;

    for (int u = 0; u < outer; u++) {
        const float *src = p + (size_t)u * inner;
        float *dst = o + (size_t)u * inner;
        normalize_unit(src, dst, inner, g, b, eps);
    }
    return FE_OK;
}

/* GroupNorm: [N, C, H, W]; group the C channels, normalize each group's
 * (C/G * H * W) extent independently. Affine params are per-channel. */
FeStatus fe_groupnorm(const FeTensor *in, const FeTensor *gamma,
                      const FeTensor *beta, FeTensor *out,
                      int groups, float eps) {
    if (!in || !out || !gamma || !beta) return FE_ERR_NULL;
    if (in->ndim < 2) return FE_ERR_SHAPE;
    int N = in->shape[0];
    int C = in->shape[1];
    if (groups < 1 || C % groups != 0) return FE_ERR_SHAPE;
    if (gamma->shape[0] != C || beta->shape[0] != C) return FE_ERR_SHAPE;
    if (in->dtype != DTYPE_FLOAT32 || out->dtype != DTYPE_FLOAT32)
        return FE_ERR_DTYPE;

    int spatial = 1;
    for (int d = 2; d < in->ndim; d++) spatial *= in->shape[d];
    int cpg = C / groups;              /* channels per group */
    int unit = cpg * spatial;          /* elements normalized together */

    const float *p = (const float *)in->data;
    float *o = (float *)out->data;
    const float *g = (const float *)gamma->data;
    const float *b = (const float *)beta->data;

    for (int n = 0; n < N; n++) {
        for (int gr = 0; gr < groups; gr++) {
            const float *src = p + ((size_t)n * C + gr * cpg) * spatial;
            float *dst = o + ((size_t)n * C + gr * cpg) * spatial;

            double sum = 0.0, sumsq = 0.0;
            for (int i = 0; i < unit; i++) {
                double v = src[i];
                sum += v; sumsq += v * v;
            }
            double mean = sum / unit;
            double var  = sumsq / unit - mean * mean;
            if (var < 0.0) var = 0.0;
            float inv_std = 1.0f / (float)sqrt(var + eps);

            for (int ch = 0; ch < cpg; ch++) {
                int ch_glob = gr * cpg + ch;
                const float *csrc = src + (size_t)ch * spatial;
                float *cdst = dst + (size_t)ch * spatial;
                for (int i = 0; i < spatial; i++)
                    cdst[i] = g[ch_glob] * ((csrc[i] - (float)mean) * inv_std) + b[ch_glob];
            }
        }
    }
    return FE_OK;
}
