# Stage 10 — SIMD

## Bottom Line Up Front

SIMD is the performance story: runtime-detected AVX2 kernels that replace naive loops behind the same `ops/` contract. The scalar path stays the default and the reference; SIMD is an optimization, never a correctness source. **Done when:** AVX2 matmul is dispatched by the engine behind CPUID and remains bit-comparable to the naive kernel within tolerance.

Ferrite already has a tiled, FMA-based AVX2 matmul with ~13.4× speedup on 256×256 and runtime detection via CPUID. Gaps: routing it into engine dispatch, remainder-column handling for arbitrary N, and SIMD variants of more ops.

## Deliverables

- AVX2 vector class
- SIMD Add
- SIMD Mul
- SIMD GEMM
- SIMD Conv
- SIMD Activation
- FMA optimization

## How to Proceed

1. **The scalar kernel is the reference.** Every SIMD kernel is validated against the naive one with a max-error check before it is reported. `bench_avx2` already does this for matmul — every new SIMD kernel gets the same treatment.
2. **Keep the same contract.** SIMD kernels take the same tensors, validate the same way, and return the same `FeStatus`. The dispatch switch should not care which implementation runs.
3. **Detect once, dispatch at run time.** `fe_cpu_has_avx2()` via CPUID decides which implementation the engine calls. The binary must run on machines without AVX2 — scalar is the default.
4. **Tiling and packing first, FMA second.** The existing matmul tiles K (`KC = 256`) and packs B as `B_tile[k][8]` so the inner loop reads 8 contiguous floats. The cache behavior is most of the speedup; FMA (`_mm256_fmadd_ps`) is the cherry on top. Replicate this structure for other ops.
5. **Handle remainders explicitly.** `N % 8 != 0` falls back to scalar. That is correct and must stay tested — an arbitrarily shaped model is the normal case, not the edge case.
6. **A vector class is sugar, not a requirement.** If an abstraction makes the kernel clearer, add it; if it obscures the intrinsics, skip it. The priority is correct, fast, readable kernels.
7. **SIMD activations are elementwise and trivial.** ReLU, sigmoid, tanh are embarrassingly parallel; a 8-wide loop with a scalar tail covers them. Do these after the GEMM/conv path, not before.
8. **SIMD Conv reuses im2col.** Conv becomes a GEMM, so SIMD GEMM makes conv fast for free. Optimize the GEMM, not the conv.

**Verify.** `tests/` has a SIMD-vs-naive equivalence test for each kernel over shapes that hit both the vector path and the remainder path. The engine test asserts the AVX2 path matches the scalar path end-to-end.

**Do not** ship a SIMD kernel that can disagree with the scalar one. If a mismatch appears, the SIMD kernel is wrong — the naive one is the reference.
