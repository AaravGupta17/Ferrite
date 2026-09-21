#!/usr/bin/env python3
"""
tools/golden_compare.py — Ferrite vs ONNX Runtime golden comparison (Phase D).

For every model in tools/golden_models/, runs the SAME seeded float32 input
through (a) ONNX Runtime on the CPU provider and (b) the Ferrite engine via
tools/run_model, then asserts the outputs match within tolerance.

This is the independent-reference correctness proof: Ferrite is a from-
first-principles runtime with no shared code with ONNX Runtime, so matching
outputs on identically-shaped models is strong evidence the kernels and the
importer agree with a mainstream implementation.

Usage:
  python3 tools/golden_compare.py --run-model build/run_model
  python3 tools/golden_compare.py --run-model build/run_model --atol 5e-4
  python3 tools/golden_compare.py --run-model build/run_model --only convbn

The harness exits non-zero if any model's output misses the tolerance or a
runtime step fails (model load, inference, I/O).
"""
import argparse
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent))
from golden_gen import OUTDIR  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
# Models that live outside the generated zoo (committed fixtures).
EXTRA_PATHS = {"acousticleaknet.onnx": REPO / "tests" / "acousticleaknet.onnx"}

BIN_MAGIC = b"FRT1"
RNG = np.random.default_rng(424242)  # fixed per session => reproducible runs

# model name -> input shape / output shape (mirrors tools/golden_gen.py).
MODELS = {
    "mlp.onnx":     ([1, 6],   [1, 4]),
    "minicnn.onnx": ([1, 1, 8, 8], [1, 8]),
    "convbn.onnx":  ([1, 1, 8, 8], [1, 4]),
    "normmlp.onnx": ([1, 8],   [1, 3]),
    "acousticleaknet.onnx": ([1, 1, 1024], [1, 3]),
}


def write_frt1(path, arr):
    arr = np.ascontiguousarray(arr, dtype="<f4")
    with open(path, "wb") as f:
        f.write(BIN_MAGIC)
        f.write(struct.pack("<ii", 1, arr.ndim))
        for d in arr.shape:
            f.write(struct.pack("<i", d))
        f.write(arr.tobytes())


def read_frt1(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        ver, ndim = struct.unpack("<ii", f.read(8))
        dims = struct.unpack(f"<{ndim}i", f.read(4 * ndim))
        data = np.frombuffer(f.read(), dtype="<f4")
    if magic != BIN_MAGIC or ver != 1:
        raise ValueError(f"{path}: bad FRT1 header")
    return data.reshape(dims)


def run_onnxruntime(model_path, x):
    sess = ort.InferenceSession(str(model_path),
                                providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    return sess.run(None, {name: x})[0]


def run_ferrite(run_model, model_path, x, out_shape):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        in_bin = tmp / "input.bin"
        out_bin = tmp / "output.bin"
        cfg = tmp / "out_dims.cfg"
        write_frt1(in_bin, x)
        cfg.write_text(" ".join(str(d) for d in out_shape))

        proc = subprocess.run(
            [run_model, str(model_path), str(in_bin), str(out_bin), str(cfg)],
            capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"  run_model failed ({proc.returncode}): "
                  f"{proc.stderr.strip()}", file=sys.stderr)
            return None
        return read_frt1(out_bin)


def compare(model_name, run_model, atol, rtol):
    model_path = EXTRA_PATHS.get(model_name, OUTDIR / model_name)
    in_shape, out_shape = MODELS[model_name]

    x = RNG.uniform(-1.0, 1.0, in_shape).astype(np.float32)

    ref = run_onnxruntime(model_path, x)
    got = run_ferrite(run_model, model_path, x, out_shape)
    if got is None:
        return False

    if ref.shape != got.shape:
        print(f"  FAIL {model_name}: shape {tuple(got.shape)} != "
              f"{tuple(ref.shape)}")
        return False

    total = ref.size
    n_close = int(np.sum(np.isclose(got, ref, atol=atol, rtol=rtol)))
    max_abs = float(np.max(np.abs(got - ref)))
    max_rel = float(np.max(np.abs(got - ref) /
                           np.maximum(np.abs(ref), np.finfo(np.float32).eps)))
    ok = n_close == total
    status = "PASS" if ok else "FAIL"
    print(f"  {status:4s} {model_name:<12} max_abs={max_abs:.3e} "
          f"max_rel={max_rel:.3e} match={n_close}/{total}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-model", required=True,
                    help="path to the built tools/run_model binary")
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument("--rtol", type=float, default=1e-4)
    ap.add_argument("--only", default=None, help="run a single model name")
    args = ap.parse_args()

    if not Path(args.run_model).exists():
        print(f"run_model binary not found: {args.run_model}", file=sys.stderr)
        return 2

    if not OUTDIR.exists():
        print(f"model zoo missing at {OUTDIR} — run tools/golden_gen.py first",
              file=sys.stderr)
        return 2

    print(f"golden comparison  (atol={args.atol:.0e}, rtol={args.rtol:.0e})")
    all_ok = True
    names = [args.only] if args.only else sorted(MODELS)
    for name in names:
        if name not in MODELS:
            print(f"unknown model {name}", file=sys.stderr)
            return 2
        all_ok &= compare(name, args.run_model, args.atol, args.rtol)

    print("PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())