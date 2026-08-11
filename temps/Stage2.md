# Stage 2 — Math Backend

## Bottom Line Up Front

The math backend is a library of correct, tested math over tensors — the building blocks operators and kernels call. No engine logic here. **Done when:** every routine has a test and the naive matmul is the reference the SIMD kernel validates against.

Ferrite's `ops/matmul.c` and `ops/activations.c` already cover matrix multiplication, dot products, and elementwise ops. Gaps: scalar/vector ops as first-class routines, reductions, random generation, and numerical-stability helpers.

## Deliverables

- Scalar operations
- Vector operations
- Matrix multiplication
- Dot product
- Reduction operators
- Elementwise operations
- Random tensor generation
- Numerical stability helpers

## How to Proceed

1. **Correctness first.** Write the naive loop before any optimization. The naive kernel is the reference implementation; never "fix" it to match an optimized one. This is how `fe_matmul_avx2` is validated today.
2. **Kernel contract everywhere.** Inputs read-only, output caller-allocated, no allocation inside, validate pointers/dtypes/shapes first. Follow `ops/ops.h` exactly.
3. **Reductions.** Implement sum, mean, max, min, argmax over a specified axis. Compute the row/column decomposition once (product of dims before/after the axis) so any rank works.
4. **Numerical stability.** The model for this is `fe_softmax`: subtract the max before `exp`. Apply the same discipline to any sum-of-exponentials, log-sum-exp, and variance (use the two-pass or Welford form). Document the choice in the header.
5. **Random generation.** Seedable, deterministic, and testable. A fixed seed must produce identical output across runs — stress tests depend on it. Do not use the system RNG without seeding control.
6. **Scalar and vector ops.** Thin wrappers over the same validated elementwise core. Prefer one generic loop over N specialized functions.
7. **Shape validation.** Every entry point checks shapes before math. A `FE_ERR_SHAPE` here stops a silent wrong-answer three layers up.

**Verify.** `tests/test_ops.c` covers each routine including edge cases: empty tensors, 1×1 matmul, float overflow in softmax, reduction over the middle axis. All green under ASan/UBSan before moving on.

**Do not** optimize before correctness. Stage 10 (SIMD) replaces these loops; Stage 2 exists so there is something true to compare against.
