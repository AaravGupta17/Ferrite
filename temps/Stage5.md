# Stage 5 — ONNX

## Bottom Line Up Front

ONNX support means reading a real `.onnx` file into a valid `FeGraph` with zero dependencies — a hand-rolled protobuf wire parser, weights copied into the arena, and a graph that has been validated and sorted. **Done when:** a real exported model loads, verifies, and runs; unsupported ops fail loudly instead of silently vanishing.

**Status: core deliverables landed (Phase 3 of the exec plan).**
- `value_info` parsing — GraphProto fields 11/12/13 (input/output/value_info) fill tensor registry shapes; `dim_param` parses to 0 (dynamic).
- Op map broadened to 20+ strings (basic math, activations, pool, norms, conv2d); per-op attributes (`alpha`, `epsilon`, `num_groups`, `num_heads`, `kernel_shape`, `strides`, `pads`) parsed explicitly and applied to `FeNode.attrs`.
- ONNX defaults applied (`epsilon=1e-5`, `leaky_relu=0.01`, `elu=1.0`, strides 1); missing required attrs (`kernel_shape`, `num_groups`) or required ONNX inputs (Gemm bias, LayerNorm/GroupNorm bias, BatchNorm 5 inputs) → `FE_ERR_SHAPE`.
- `fe_onnx_load` now runs `fe_graph_topo_sort` **and** `fe_graph_validate`.
- Tests: `tests/test_onnx.c` covers the tiny MLP, unsupported-op failure, `value_info` shape fill, pool/norm attribute mapping, and a malformed-length parser failure.

Open: batch normalization `epsilon`/`momentum` attrs still need the packed-float path verified against a `float_data` tensor; full shape inference belongs in Stage 12 — `value_info` only fills shapes the file declares.

## Deliverables

- Read ONNX protobuf
- Parse tensors
- Parse weights
- Parse nodes
- Parse attributes
- Build graph
- Verify graph
- Support common opsets

## How to Proceed

1. **Protobuf is a wire format, not a library.** Keep the varint/tag/skip/bytes primitives (`importer/onnx.h`) as the only protobuf surface. Extend with the smallest new primitive you need, test it in isolation first.
2. **Skip fields you do not need; fail on structures you do.** The parser already skips `value_info`. Change the policy: when a required structure is malformed, return an error — never guess.
3. **Weights go straight to the arena.** Initializer raw_data is `memcpy`'d into the weight arena exactly once, at load. No copies, no per-tensor mallocs. This is the existing contract; preserve it.
4. **Parse attributes explicitly.** `attr` proto fields carry conv strides, pads, axes. Map each ONNX attribute to the `FeNode.attrs` union field for that op. Unknown attributes on a supported op are a warning; a missing required attribute is an error.
5. **Deduplicate tensors by name.** `find_or_add_tensor` must stay the single registration path so every node referencing the same name shares one graph entry.
6. **Verify after build.** Run the Stage 4 validation on every loaded graph: index ranges, single producer per tensor, dtype consistency, and a successful topo sort. The importer returns `FE_OK` only after all of it passes.
7. **Opset policy: fail loudly.** The roadmap rule is explicit — silently dropping unsupported ops hides bugs. Log the op and return `FE_ERR_SHAPE`. Depth over breadth.
8. **Shape inference belongs here or in Stage 12.** For now, parse `value_info` when present and fill in shapes; a full propagation pass is Stage 12 (Graph Optimization). Do not block on it.

**Verify.** `tests/test_onnx.c` loads `tests/tiny_mlp.onnx` end-to-end and asserts node count, tensor count, and that weights landed in the arena. Add a negative test: a file with an unsupported op must fail loudly.

**Do not** pull in a protobuf library or a parser generator. Hand-rolled is the point of the project, and the existing parser is already the foundation.
