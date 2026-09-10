// tools/bench.h — shared benchmark helpers (Stage 8).
#ifndef FERRITE_BENCH_H
#define FERRITE_BENCH_H

/*
 * Shared timing framework for performance tools.
 *
 * Pattern: warm up (cache + branch predictor), then time the mean over
 * `runs`. Prefer best-of-many for sub-millisecond kernels; mean is fine
 * for end-to-end inference which dwarfs timer overhead.
 */
double fe_bench_ms(void);                 /* monotonic wall clock, ms */
double fe_bench_mean(int runs, void (*fn)(void *), void *arg);
double fe_bench_gflops(double ms, double n_flops);  /* GFLOPS from ms */
void   fe_bench_header(const char *name);/* title + CPU/AVX2/build line */

#endif // FERRITE_BENCH_H