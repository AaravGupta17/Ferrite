# Stage 7 — Testing

## Bottom Line Up Front

Testing is how Ferrite stays honest. Every subsystem has a test binary built under `-fsanitize=address,undefined`; correctness claims are only as strong as the tests behind them. **Done when:** unit, regression, stress, and fuzz tests all pass, and a code-coverage run shows no untested branch in the core path.

Ferrite already follows the one-test-per-subsystem model. This stage deepens it: property and tolerance tests, stress runs, and fuzzing the parser and kernels.

## Deliverables

- Tensor tests
- Operator tests
- Graph tests
- Parser tests
- Runtime tests
- Regression tests
- Stress tests
- Fuzz testing

## How to Proceed

1. **Every public function has a test.** Not a smoke call — an assertion on a hand-computed value. For kernels, that means small shapes where you know the answer by hand, plus a tolerance-based comparison against a reference.
2. **Reference over reimplementation.** The naive matmul is the reference for the AVX2 kernel. Extend that pattern: test each optimized kernel against the naive one, and the whole runtime against hand-built graphs. Never test a function only against itself.
3. **Tensor tests cover the invariants.** Contiguity, ownership (`owns_data`), view-vs-copy behavior, stride math. A `stride == 0` broadcast view must behave identically to its materialized copy.
4. **Parser tests include negatives.** Fuzzing the protobuf reader is a Stage 7 duty, not optional hardening. Truncated files, bad wire types, oversized lengths must return errors — never read out of bounds.
5. **Regression tests are commits, not chores.** When a bug is fixed, add the failing case to the suite in the same commit. It is the documented rule: code change and its test land together.
6. **Stress tests chase real limits.** Fixed capacities are 512 nodes / 1024 tensors / 8 dims. Run graphs that hit the edge of each: 512 nodes, 1024 tensors, an 8-dim tensor, `N % 8 != 0` matmuls for the AVX2 remainder path.
7. **Fuzz strategically.** Two targets matter: the ONNX parser (feed mutated bytes) and kernels (feed shape/dtype combinations). Use whatever harness is available; a hand-rolled random-input loop under ASan beats no fuzzing.
8. **Measure coverage, then close gaps.** Add a coverage flag to the build, look at the report, and write tests for what is red. Stop at the core path; 100% of the IR is not the goal.

**Verify.** Every test binary runs clean under ASan/UBSan. A regression run of all targets is one command. Failing loudly is correct behavior: an unimplemented op returning an error is a passing test, not a failure.

**Do not** weaken assertions to make tests pass. If a test is flaky or wrong, the code or the reference is wrong — find out which.
