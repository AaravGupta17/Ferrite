#!/usr/bin/env python3
"""
tools/golden_gen.py — build the golden ONNX model zoo (Phase D).

Writes small, fully-static, float32 .onnx models into tools/golden_models/,
generated with the `onnx` Python package and comprising ONLY ops the Ferrite
importer maps. Every Gemm uses ONNX default attributes (alpha=1, beta=1,
transA=0, transB=0) with an explicit C-bias input, so the importer's Gemm
-> Linear path is used with no warning.

Models (see golden_compare.py for the reference/assertion harness):
  mlp.onnx        Gemm->Relu->Gemm->Relu->Gemm->Softmax
  minicnn.onnx    Conv->Relu->AveragePool->Flatten->Gemm->Relu->Gemm->Softmax
  convbn.onnx     Conv->BatchNormalization->Relu->Flatten->Gemm->Softmax
                  (exercises the load-time Conv+BN fusion pass)
  normmlp.onnx    Gemm->LayerNormalization->LeakyRelu->Gemm->Sigmoid

Run: python3 tools/golden_gen.py            (regenerates the zoo in place)
"""
import os
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUTDIR = Path(__file__).resolve().parent / "golden_models"
RNG = np.random.default_rng(20260919)


def _initializer(name, arr):
    return numpy_helper.from_array(np.asarray(arr, dtype=np.float32), name)


def _value_info(name, shape):
    return helper.make_tensor_value_info(
        name, TensorProto.FLOAT, [int(d) for d in shape])


def _save(model, name):
    model.producer_name = "ferrite-golden-gen"
    model.ir_version = 8
    out = OUTDIR / name
    onnx.checker.check_model(model)
    onnx.save(model, out)
    print(f"wrote {out}  ({out.stat().st_size} bytes)")


def _graph(model_name, inputs_in, outputs_in, nodes, initializers, value_infos):
    graph = helper.make_graph(
        nodes,
        model_name,
        inputs=[inputs_in],
        outputs=[outputs_in],
        initializer=initializers,
        value_info=value_infos,
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)])


def mlp():
    W1 = RNG.normal(0, 0.1, (6, 16))
    b1 = RNG.normal(0, 0.1, (16,))
    W2 = RNG.normal(0, 0.1, (16, 16))
    b2 = RNG.normal(0, 0.1, (16,))
    W3 = RNG.normal(0, 0.1, (16, 4))
    b3 = RNG.normal(0, 0.1, (4,))

    n1 = helper.make_node("Gemm", ["in", "w1", "b1"], ["h1"], name="g1")
    n2 = helper.make_node("Relu", ["h1"], ["r1"], name="r1")
    n3 = helper.make_node("Gemm", ["r1", "w2", "b2"], ["h2"], name="g2")
    n4 = helper.make_node("Relu", ["h2"], ["r2"], name="r2")
    n5 = helper.make_node("Gemm", ["r2", "w3", "b3"], ["lg"], name="g3")
    n6 = helper.make_node("Softmax", ["lg"], ["out"], axis=-1, name="sm")

    init = [
        _initializer("w1", W1), _initializer("b1", b1),
        _initializer("w2", W2), _initializer("b2", b2),
        _initializer("w3", W3), _initializer("b3", b3),
    ]
    vi = [
        _value_info("h1", [1, 16]), _value_info("r1", [1, 16]),
        _value_info("h2", [1, 16]), _value_info("r2", [1, 16]),
        _value_info("lg", [1, 4]),
    ]
    _save(_graph("mlp", _value_info("in", [1, 6]),
                 _value_info("out", [1, 4]),
                 [n1, n2, n3, n4, n5, n6], init, vi), "mlp.onnx")


def minicnn():
    Wc = RNG.normal(0, 0.1, (4, 1, 2, 2))
    bc = RNG.normal(0, 0.1, (4,))
    Wf = RNG.normal(0, 0.1, (144, 16))
    bf = RNG.normal(0, 0.1, (16,))
    Wo = RNG.normal(0, 0.1, (16, 8))
    bo = RNG.normal(0, 0.1, (8,))

    n1 = helper.make_node("Conv", ["in", "wc", "bc"], ["c1"],
                          kernel_shape=[2, 2], strides=[1, 1],
                          pads=[0, 0, 0, 0], name="conv")
    n2 = helper.make_node("Relu", ["c1"], ["r1"], name="relu")
    n3 = helper.make_node("AveragePool", ["r1"], ["p1"],
                          kernel_shape=[2, 2], strides=[1, 1], name="avgpool")
    n4 = helper.make_node("Flatten", ["p1"], ["f1"], axis=1, name="flatten")
    n5 = helper.make_node("Gemm", ["f1", "wf", "bf"], ["h1"], name="g1")
    n6 = helper.make_node("Relu", ["h1"], ["r2"], name="relu2")
    n7 = helper.make_node("Gemm", ["r2", "wo", "bo"], ["lg"], name="g2")
    n8 = helper.make_node("Softmax", ["lg"], ["out"], axis=-1, name="sm")

    init = [
        _initializer("wc", Wc), _initializer("bc", bc),
        _initializer("wf", Wf), _initializer("bf", bf),
        _initializer("wo", Wo), _initializer("bo", bo),
    ]
    vi = [
        _value_info("c1", [1, 4, 7, 7]), _value_info("r1", [1, 4, 7, 7]),
        _value_info("p1", [1, 4, 6, 6]), _value_info("f1", [1, 144]),
        _value_info("h1", [1, 16]), _value_info("r2", [1, 16]),
        _value_info("lg", [1, 8]),
    ]
    _save(_graph("minicnn", _value_info("in", [1, 1, 8, 8]),
                 _value_info("out", [1, 8]),
                 [n1, n2, n3, n4, n5, n6, n7, n8], init, vi),
          "minicnn.onnx")


def convbn():
    Wc = RNG.normal(0, 0.1, (2, 1, 2, 2))
    bc = RNG.normal(0, 0.1, (2,))
    scale = RNG.normal(1.0, 0.05, (2,))
    beta = RNG.normal(0, 0.05, (2,))
    mean = RNG.normal(0, 0.05, (2,))
    var = RNG.uniform(0.5, 1.5, (2,))
    Wf = RNG.normal(0, 0.1, (98, 4))
    bf = RNG.normal(0, 0.1, (4,))

    n1 = helper.make_node("Conv", ["in", "wc", "bc"], ["c1"],
                          kernel_shape=[2, 2], strides=[1, 1],
                          pads=[0, 0, 0, 0], name="conv")
    n2 = helper.make_node("BatchNormalization",
                          ["c1", "scale", "beta", "mean", "var"], ["bn1"],
                          epsilon=1e-5, name="bn")
    n3 = helper.make_node("Relu", ["bn1"], ["r1"], name="relu")
    n4 = helper.make_node("Flatten", ["r1"], ["f1"], axis=1, name="flatten")
    n5 = helper.make_node("Gemm", ["f1", "wf", "bf"], ["lg"], name="g1")
    n6 = helper.make_node("Softmax", ["lg"], ["out"], axis=-1, name="sm")

    init = [
        _initializer("wc", Wc), _initializer("bc", bc),
        _initializer("scale", scale), _initializer("beta", beta),
        _initializer("mean", mean), _initializer("var", var),
        _initializer("wf", Wf), _initializer("bf", bf),
    ]
    vi = [
        _value_info("c1", [1, 2, 7, 7]), _value_info("bn1", [1, 2, 7, 7]),
        _value_info("r1", [1, 2, 7, 7]), _value_info("f1", [1, 98]),
        _value_info("lg", [1, 4]),
    ]
    _save(_graph("convbn", _value_info("in", [1, 1, 8, 8]),
                 _value_info("out", [1, 4]),
                 [n1, n2, n3, n4, n5, n6], init, vi),
          "convbn.onnx")


def normmlp():
    W1 = RNG.normal(0, 0.1, (8, 12))
    b1 = RNG.normal(0, 0.1, (12,))
    scale = RNG.normal(1.0, 0.05, (12,))
    bias = RNG.normal(0, 0.05, (12,))
    W2 = RNG.normal(0, 0.1, (12, 3))
    b2 = RNG.normal(0, 0.1, (3,))

    n1 = helper.make_node("Gemm", ["in", "w1", "b1"], ["h1"], name="g1")
    n2 = helper.make_node("LayerNormalization", ["h1", "scale", "bias"],
                          ["ln1"], axis=-1, epsilon=1e-5, name="layernorm")
    n3 = helper.make_node("LeakyRelu", ["ln1"], ["lr1"], alpha=0.05,
                          name="leaky")
    n4 = helper.make_node("Gemm", ["lr1", "w2", "b2"], ["sig"], name="g2")
    n5 = helper.make_node("Sigmoid", ["sig"], ["out"], name="sigmoid")

    init = [
        _initializer("w1", W1), _initializer("b1", b1),
        _initializer("scale", scale), _initializer("bias", bias),
        _initializer("w2", W2), _initializer("b2", b2),
    ]
    vi = [
        _value_info("h1", [1, 12]), _value_info("ln1", [1, 12]),
        _value_info("lr1", [1, 12]), _value_info("sig", [1, 3]),
    ]
    _save(_graph("normmlp", _value_info("in", [1, 8]),
                 _value_info("out", [1, 3]),
                 [n1, n2, n3, n4, n5], init, vi),
          "normmlp.onnx")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    mlp()
    minicnn()
    convbn()
    normmlp()


if __name__ == "__main__":
    main()