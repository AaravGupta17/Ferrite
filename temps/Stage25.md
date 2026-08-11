# Stage 25 — Ecosystem

## Bottom Line Up Front

The ecosystem is everything around the runtime that makes it usable in practice: pretrained models, training support, tooling integrations, and a package manager. **Done when:** someone can install Ferrite, load a pretrained model from the Model Zoo, and run it — with the training pipeline feeding the zoo.

This stage assumes the core is a finished product. It is the "company around the product" stage, and none of it is required for Ferrite to be useful.

## Deliverables

- Model Zoo
- Pretrained models
- Training support
- Custom compiler
- IDE extension
- VSCode integration
- Package manager
- Cloud inference

## How to Proceed

1. **The Model Zoo is the single most valuable item.** A curated set of small ONNX models (MNIST MLP, a tiny CNN, a sequence model) with license, source (training script), accuracy, and export command for each. The demo model (roadmap P2) is the first entry. The zoo is what makes "clone and run something real" true.
2. **Pretrained models are the zoo's contents, not a feature.** Each model must include: the ONNX file, the calibration data if quantized (Stage 15), expected outputs for a golden-tensor test (roadmap Phase 5), and provenance. A model without provenance is untrustworthy.
3. **Training support means a converter, not a trainer.** Ferrite does not train. The realistic item is a PyTorch→ONNX export helper and a validation script that checks a Ferrite build reproduces the PyTorch output. Training itself stays in Python.
4. **The custom compiler is Stage 14's work product as a standalone tool.** A `ferrite compile model.onnx -o model.fer` step that runs the full pass pipeline and emits a deployable artifact. It only exists if the compiler work shipped.
5. **IDE extension (VSCode) is syntax + run helpers.** Highlight `.onnx` model descriptions? No — the honest scope is a task runner: build, test, run a model, view the profiler table. `tasks.json`/`launch.json` committed to the repo is 90% of the value.
6. **Package manager = a release channel.** A vcpkg/conan/Homebrew recipe that installs the library, headers, and CLI. It is packaging work on top of the Stage 20 release process. Add it when the C API is stable and someone asks for it.
7. **Cloud inference is a deployment recipe, not a service.** A Dockerfile + example that serves a model over HTTP with the CLI or bindings. Cloud *service* hosting is a business decision, out of scope here.
8. **Every ecosystem item links back to a core deliverable.** If a zoo model cannot run, that is an engine bug to fix, not a zoo problem to paper over.

**Verify.** The zoo's demo model: install → load → run → correct output, matching the committed golden tensor. Each new model entry follows the same checklist and passes the validation script.

**Do not** build the ecosystem on top of a demo that is not finished. The ecosystem amplifies the core — if the core is shaky, it amplifies the wrong things.
