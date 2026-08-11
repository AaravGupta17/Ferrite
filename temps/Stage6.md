# Stage 6 — Execution Engine

## Bottom Line Up Front

The engine is where the graph runs: it walks topological order, dispatches each node to a kernel, and manages activation memory. It is a thin, boring layer — the hard math lives in `ops/`. **Done when:** a multi-op graph runs end-to-end, every dispatched op is tested through the engine, and errors propagate without crashing.

Ferrite's `runtime/engine.c` already runs this pipeline for six ops. Gaps: dispatching `CONV1D`/`BATCHNORM`, consuming the planner's `FePlan`, and a formal runtime context + intermediate tensor manager.

## Deliverables

- Executor
- Operator dispatcher
- Runtime context
- Intermediate tensor management
- Error propagation
- Graph execution
- Batch execution

## How to Proceed

1. **Executor walks `topo_order`, nothing else.** The engine reads the graph's precomputed order and dispatches in a `switch` on `node->op`. No re-sorting, no analysis at run time.
2. **Dispatcher is exhaustive.** Every `FeOpType` has a case. `CONV1D` and `BATCHNORM` are the two known gaps — add their kernels (Stage 3) and their cases together. The default branch prints the op and returns `FE_ERR_SHAPE`; it must be dead code, not a resting state.
3. **Runtime context.** Keep `FeRuntime` as the single context struct: graph pointer, weight arena, activation arena, optional profiler. It already exists; extend it, do not scatter globals.
4. **Intermediate tensors come from the arena.** Reset the activation arena, bump-allocate every non-weight tensor, bind the caller's input, run, copy the output. That is the current pipeline — and the correct one. This is Stage 13's job to optimize.
5. **Error propagation.** A kernel failure must unwind the run and return a status, leaving the runtime reusable for the next call. Reset state on entry, never mid-call. The caller's output tensor is untouched on failure.
6. **Bind input by contract.** The input node's output tensor is a view of the caller's buffer; the caller owns it. Do not copy unless the graph needs contiguity.
7. **Batch execution is a loop over runs.** Bind a fresh input slice, run, read the output slice, repeat — reusing the same runtime and arenas. Implement it as a helper over `fe_runtime_run`, not a new code path.
8. **Profile from the start.** Every dispatch already gets wrapped in `fe_profiler_now_ns()` when a profiler is attached. Keep it; the report is how you prove Stage 8 numbers.

**Verify.** `tests/test_engine.c` is the model: a hand-built MLP (Linear → ReLU → Linear → Softmax) with known weights and an exact expected output. Add a conv1d graph and a batchnorm graph as they land, each with a hand-computed expected result.

**Do not** put math in the engine. If a dispatch case needs more than a kernel call and a status check, the logic belongs in `ops/`.
