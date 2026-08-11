# Stage 3 — Operator Library

## Bottom Line Up Front

The operator library is the family of kernels a model can be built from, grouped by family: basic math, activations, linear algebra, CNN, and sequence ops. **Done when:** each op is a tested kernel behind the `ops/` contract, dispatched by the engine, and never silently skipped.

Ferrite already ships the basic, activation, and linear-algebra kernels it needs: add, matmul, linear, relu, softmax, bias_add, conv1d. The gaps are the wider activation set, 2D conv, the norm layers, and all sequence ops. Depth over breadth: implement what a real model needs, in family order.

## Deliverables

**Basic:** Add, Sub, Mul, Div, Neg, Exp, Log, Pow
**Neural Network:** ReLU, Sigmoid, Tanh, GELU, Softmax, Leaky ReLU, ELU, Swish
**Linear Algebra:** MatMul, GEMM, Transpose
**CNN:** Conv2D, MaxPool, AvgPool, BatchNorm, LayerNorm, GroupNorm
**Sequence:** Attention, MultiHead Attention, Embedding, Positional Encoding

## How to Proceed

1. **Family order, not alphabetical.** Basic math first (Stage 2 wraps these), then activations, then linear algebra, then CNN, then sequence. Each family builds on the previous one's tests and idioms.
2. **One kernel, one file, one test.** Match the existing pattern: `ops/<op>.c`, one test case in `tests/test_*.c`. A new op is a new file, not a case added to an unrelated one. Conv2d, norms, and attention each deserve their own test binary.
3. **Attribute-driven, like the IR.** Ops with parameters (softmax axis, conv stride/pad, norm eps) carry them in the node's `attrs` union, not in hidden globals. See `graph/graph.h` for the pattern.
4. **GEMM before Conv.** Implement GEMM as the general form (with transposition flags and alpha/beta), then express conv via im2col → GEMM, exactly as `fe_conv1d` does. Reuse the proven path.
5. **Norms share a skeleton.** BatchNorm, LayerNorm, GroupNorm differ only in which axes reduce. Implement one normalized-reduction helper and three thin wrappers over it.
6. **Sequence ops last.** Attention needs matmul, softmax, and a scale — all already built. MultiHead attention is reshaping + batching over heads, not new math. Embedding is a gather with weights owned by the arena.
7. **Numerical stability by default.** Softmax subtracts max; GELU uses the tanh approximation or the exact erf with a documented choice; norms use a small epsilon to avoid divide-by-zero.
8. **Update the dispatcher with each op.** Every new kernel gets a `case` in `runtime/engine.c`. An op in the enum without a dispatch case is a known bug, not a TODO.

**Verify.** Each op passes its test under ASan/UBSan, and at least one multi-op graph test runs end-to-end through the engine (like `tests/test_engine.c`).

**Do not** add an op "because it exists in ONNX." Add ops the current goal and demo actually need. Depth over breadth is the project rule.
