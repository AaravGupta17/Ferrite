# Stage 12 — Graph Optimization

## Bottom Line Up Front

Graph optimization rewrites the graph to do less work without changing its output: fold constants, drop dead nodes, fuse ops, and infer shapes. Every pass is validated by an output-equivalence test against the unoptimized graph. **Done when:** a test model runs with fewer nodes and identical results, with each pass measured.

Ferrite has no optimization passes yet. This stage sits after the graph IR (Stage 4) is solid and before the compiler features (Stage 14) that consume optimized graphs.

## Deliverables

- Constant folding
- Dead node elimination
- Dead tensor elimination
- Operator fusion
- Graph simplification
- Common subexpression elimination
- Shape inference
- Constant propagation

## How to Proceed

1. **Shape inference first.** Every other pass needs shapes. Propagate shapes forward from graph inputs through each supported op (matmul, linear, relu, softmax, conv, norms). This also closes the importer's value_info gap (Stage 5). Output shape rules live in one table keyed by op.
2. **Constant folding for weights.** Subgraphs whose inputs are all weights (e.g., `Transpose(weight)`, `Reshape(weight)`) can be computed once at load. Fold them during import, not at run time, and store the result in the weight arena.
3. **Dead elimination is bookkeeping.** A node whose outputs are never consumed is dead; a tensor no node references is dead. Run both until fixpoint — deleting a node can make another node dead. The lifetime analysis (Stage 9) gives you the consumer sets.
4. **Constant propagation is folding's general form.** Any input known constant at load time (weights, folded results) lets the pass attempt to fold the node. Fall back to runtime execution when the inputs are not constant.
5. **Operator fusion is where the real wins are.** The canonical fusions: `Linear` is already fused (`matmul` + `bias_add`), and `BatchNorm` folds into preceding `Conv`/`Linear` weights at inference time (conv-batchnorm fusion is a classic, big win). Emit fused ops only when the IR has a node type for them.
6. **CSE after the rest.** Find nodes with identical op, inputs, and attributes; keep one, rewrite consumers. Cheap on small graphs; still worth doing once.
7. **Graph simplification is cleanup.** Remove identity reshapes and redundant transposes (a transpose whose inverse immediately follows). Match simple, provable patterns; never guess.
8. **Every pass is a function with a test.** `FeGraph *fe_opt_fold_constants(g)` style — in, out, no mutation of the input. The equivalence test runs both graphs through the engine and compares outputs within tolerance, then asserts the optimized graph has fewer nodes/ops.

**Verify.** `tests/test_opt.c`: one graph per pass with a hand-computed optimized form, plus an end-to-end equivalence test. A fused conv-batchnorm graph must produce identical results to the unfused one.

**Do not** optimize correctness away. A pass that changes output is not an optimization — it is a bug. Equivalence is the acceptance test for every pass.
