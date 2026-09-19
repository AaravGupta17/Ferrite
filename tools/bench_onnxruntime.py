#!/usr/bin/env python3
"""
tools/bench_onnxruntime.py — external baseline for the Ferrite vs. ONNX
Runtime comparison.

Loads the same model Ferrite's tools/bench_model.c uses
(tests/acousticleaknet.onnx), builds the identical synthetic input (a
440 Hz sine tone sampled at 8000 Hz, 1024 samples — see bench_model.c),
and times inference the same way: 1 discarded warmup run, then the mean
of 20 timed runs.

Reports two numbers:
  - single-thread ORT (intra_op_num_threads=1): the fair, apples-to-apples
    comparison against Ferrite, which does not parallelize this path.
  - default ORT (all cores): what a real deployment would actually see,
    reported separately and labeled as such — not used for the headline
    claim, since it isn't comparing the same thing.

Also prints ORT's raw output tensor so it can be diffed against Ferrite's
to confirm numerical agreement, not just speed.
"""
import time
import numpy as np
import onnxruntime as ort

MODEL_PATH = "tests/acousticleaknet.onnx"
RUNS = 20


def make_input():
    n = 1024
    sr = 8000.0
    freq = 440.0
    t = np.arange(n, dtype=np.float32)
    sig = np.sin(2.0 * np.pi * freq * t / sr).astype(np.float32)
    return sig.reshape(1, 1, n)


def bench(sess, input_name, x):
    # warmup (discarded, matches bench_model.c's correctness-check run)
    sess.run(None, {input_name: x})

    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        out = sess.run(None, {input_name: x})
        times.append((time.perf_counter() - t0) * 1000.0)
    mean_ms = sum(times) / len(times)
    return mean_ms, out[0]


def run(label, intra_threads):
    so = ort.SessionOptions()
    so.intra_op_num_threads = intra_threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(MODEL_PATH, sess_options=so,
                                providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    x = make_input()
    mean_ms, out = bench(sess, input_name, x)
    print(f"{label}:")
    print(f"  Inference: {mean_ms:.3f} ms/run (mean of {RUNS})")
    print(f"  Throughput: {1000.0 / mean_ms:.1f} inf/s")
    print(f"  Output: {out.tolist()}")
    print()
    return mean_ms, out


def main():
    print(f"ONNX Runtime version: {ort.__version__}")
    print(f"Model: {MODEL_PATH}\n")

    # Graph optimizations disabled on both runs above: ORT's graph
    # optimizer (constant folding, op fusion, etc.) is a different, much
    # more mature version of exactly what Ferrite's optim/ does — leaving
    # it on would make this a "Ferrite kernels vs. ORT kernels + ORT's
    # optimizer" comparison, not a fair kernel-for-kernel one.
    run("ONNX Runtime — single-thread (fair comparison)", 1)
    run("ONNX Runtime — default (all cores, for context only)", 0)


if __name__ == "__main__":
    main()