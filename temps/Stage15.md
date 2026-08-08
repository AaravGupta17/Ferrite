# Stage 15 — Quantization

## Bottom Line Up Front

Quantization trades a small accuracy loss for a large size and speed win: weights shrink 4×, and int arithmetic can beat float. Symmetric, zero-point-free schemes are the safe start. **Done when:** a quantized model runs through the engine with measured accuracy delta, size reduction, and speedup — all reported, not assumed.

Ferrite already has symmetric per-tensor INT8 quantization (`quantization/`). This stage extends it to per-channel scales and more dtypes, and — critically — integrates it into a real model run.

## Deliverables

- INT8
- INT16
- FP16
- BF16
- Calibration
- Quantized kernels

## How to Proceed

1. **Per-tensor is done; per-channel is next.** The known gap (roadmap, `explain.md`). Per-channel scales on conv/linear weights recover most of the accuracy lost by per-tensor. Extend `FeQuantParams` to carry per-channel scales, and use them in `fe_matmul_int8`. This is the recommended P4 feature.
2. **INT8 stays the workhorse.** int8→int32 accumulation is the proven path and already implemented. Nail per-channel INT8, validate it end-to-end, measure it — before touching INT16/FP16.
3. **INT16 for accuracy-critical paths.** Double the accumulator/scale precision when INT8 accuracy is insufficient. It is a parameter change to the quantized GEMM, not a new design.
4. **FP16 and BF16 are memory formats, not compute wins (on CPU).** They halve weight/activation memory and are the natural bridge to GPU (Stage 16). Implement storage + conversion first; SIMD float kernels can often consume FP16 after a load-time convert.
5. **Calibration chooses the scale.** Collect activation ranges by running a calibration set through the float model, then set scales from observed stats (min/max or percentile) — not from a single tensor's max. Per-tensor `max(|x|)/127` is the naive baseline; calibration is the improvement.
6. **Quantized kernels mirror the float contract.** Same tensors, same `FeStatus`, same validation. The dispatch picks the int kernel when the graph is quantized. A graph is either all-float or all-quantized — no mixed dispatch.
7. **Weights quantize at load; activations quantize per-run.** Weights convert once and store in the weight arena. Activations need scale factors computed from calibration, then quantize/dequantize in the run. Keep float as the default and INT8 as a run option (roadmap Phase 4).
8. **Report all three numbers.** Accuracy delta vs. float (on the demo model), model size after quant, and latency/throughput. The README table needs all three or the feature is not credible.

**Verify.** `tests/test_quant.c` covers the round-trip. Add an engine test running the demo model float vs. INT8 and asserting the output is within the accuracy budget measured during calibration.

**Do not** claim a speedup or an accuracy number without running the benchmark. Quantization invites hand-waving; the stage's job is to eliminate it.
