# Contributing to Ferrite

Thanks for looking. Ferrite is small on purpose, and the rules below are what keep it
that way.

## The one rule that matters

**Zero dependencies.** Ferrite links libc and nothing else. A pull request that adds a
third-party library — however good — will be declined, because the absence of
dependencies *is* the project. Python packages used by developer tooling
(`onnxruntime`, `torch`) are fine; they never enter the runtime.

## Build and test

```sh
cmake -S . -B build -G Ninja                  # GCC or Clang; MSVC is rejected
cmake --build build
ctest --test-dir build --output-on-failure    # 19 entries, all must pass
```

Sanitizers are on by default and probed for availability. If your toolchain lacks
them, configure prints `ASan/UBSan requested but unavailable` and continues — your
change may still fail CI's mandatory sanitize leg, so check there before assuming
you're green.

Opt-in targets are `EXCLUDE_FROM_ALL` and must be named:
`demo_mnist`, `mnist_eval`, `demo`, `bench_avx2`, `bench_model`, `run_model`,
`fuzz_onnx_fuzzer`.

## Adding a test

There is no test framework and no assertion macro. The pattern is a `static void
test_thing(void)` using bare `assert()`, ending in `printf("PASS test_thing\n")`, called
from `main()`. Headers are included by relative path. Fixtures resolve from the source
root, which is why CTest sets `WORKING_DIRECTORY`.

Registering `test_foo` takes **three** edits in `CMakeLists.txt`: the `add_executable`,
a `target_link_libraries` line, and adding it to the `foreach(t ...)` list. Miss the
third and the binary builds but never runs.

## What a change must include

1. **A test.** New behaviour gets a case in the matching `test_*` binary; a new
   subsystem gets its own binary.
2. **Updated docs, in the same commit.** Behaviour or structure changes
   `docs/guide.md`; scope changes `docs/roadmap.md`; build or usage changes
   `README.md`. A change is not done until its documentation is done.
3. **Header comments** for any new public function — what it does, what it returns,
   and any constraint on alignment, contiguity, or ownership.
4. **Real numbers.** Every figure written into a doc must be reproducible by a command
   printed next to it. No estimates presented as measurements.

## Invariants not to break

- **Kernels never allocate and never log.** `fe_conv1d` is the sole allocating
  exception, and it frees its scratch.
- **Views over copies.** `fe_tensor_transpose` and `fe_tensor_reshape` never copy.
- **The naive kernel is the reference** that validates the SIMD one. Don't optimize it
  away.
- **Fail loudly.** An unsupported op, an over-rank tensor, a truncated field — all
  return an error naming the problem. Nothing is silently dropped, truncated, or
  skipped. This is the invariant most likely to be broken by accident.

## Commits

One subsystem per commit, scoped: `fix(importer): short summary`. Explain *why*, not
what — the diff already says what. No AI attribution trailers.

## Reporting a model that won't load

Most valuable bug report this project can get. Please include the ONNX opset, the
operator that failed, the exact stderr message, and the model if you can share it.
Ferrite prints the op name it could not map, so paste that line verbatim.
