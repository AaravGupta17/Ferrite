# Stage 18 — Developer Tools

## Bottom Line Up Front

Developer tools make Ferrite inspectable: a CLI to load and run models, a profiler, a graph visualizer, and bindings so other languages can call in. They are convenience on top of the runtime — none of them change the core. **Done when:** a model can be loaded, run, profiled, and inspected from one command.

Ferrite already has a profiler (`tools/`) and debug printing on tensors and graphs. This stage grows those into a real tool surface.

## Deliverables

- CLI
- Python bindings
- C API
- Model inspector
- Profiler
- Graph visualizer
- Debugger

## How to Proceed

1. **CLI first — it exercises every other tool.** `ferrite run model.onnx input.bin` → print output, latency, and memory. `ferrite inspect model.onnx` → dump nodes, tensors, shapes, weights summary. The CLI is the reference way to drive the runtime by hand.
2. **The C API is the runtime's own headers, stabilized.** `core/`, `graph/`, `runtime/`, `importer/` already are the C API. Add `FE_FERRITE_VERSION` and export markers, and freeze the public surface in a single header (or a documented list) so bindings can rely on it.
3. **Python bindings come after the C API stabilizes.** A ctypes or CFFI wrapper needs no build complexity: load the runtime library, pass buffers as `numpy` arrays, get results back. Python is the tooling bridge to PyTorch/ONNX Runtime comparisons (Stage 8), so it earns its place.
4. **Model inspector prints the truth.** Reuse `fe_graph_print` + `fe_plan_print` + `fe_profiler_print`: nodes in topo order, tensor shapes/dtypes, memory offsets, per-op timing. One report, all three subsystems.
5. **Profiler already exists — expose it.** The report format (op, calls, total/avg ms, % time) is good. Route it through the CLI so `ferrite profile model.onnx` prints the table. Sort by total time as it already does.
6. **Graph visualizer is a debug export.** Emit DOT (Stage 4) from the CLI: `ferrite graphviz model.onnx > graph.dot`. Render with an external tool; do not build a renderer.
7. **Debugger is last and minimal.** A `--trace` flag that prints each dispatched node with input/output tensor summaries is 90% of debugging value. A step-through interactive debugger is out of scope unless a consumer asks for it.
8. **Every tool is a separate `tools/` file with a test.** Same one-subsystem-per-commit rule. The CLI must not link against more than the runtime — tools sit on top.

**Verify.** `tests/test_cli.c` runs the CLI binary as a subprocess against `tiny_mlp.onnx` and asserts exit code and output content. The profiler table and inspector dump are asserted for expected lines.

**Do not** let tools grow private copy-paste logic. A tool that needs something the runtime does not expose is a signal to fix the runtime API, not to fork it.
