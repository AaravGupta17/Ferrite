/*
 * fuzz/fuzz_onnx.h — shared entry for the ONNX-parser fuzz targets (Section 1.3).
 *
 * One implementation (`fuzz_onnx_input`, in fuzz_onnx.c) is wrapped two ways:
 *   - fuzz_onnx_libfuzzer.c bridges it to `LLVMFuzzerTestOneInput` for clang's
 *     libFuzzer (`-fsanitize=fuzzer,address,undefined`), and
 *   - fuzz_main.c provides a standalone `main` that replays every file named
 *     on the command line through the same function, so non-Clang toolchains
 *     get the identical code path (used by CTest: `fuzz_runner` over fuzz/corpus
 *     + the tests/ ONNX fixtures).
 *
 * The input bytes are written to a temp file and handed to fe_onnx_load, the
 * real device-facing entry point. The contract under test: any byte string
 * loads cleanly or fails loudly (a FeStatus error, not a crash, hang, or
 * sanitizer trip). This complements the fixed-corpus `test_parser_fuzz` in
 * tests/test_onnx.c — that one is the deterministic regression; this harness
 * is what "goes deeper" under libFuzzer's mutation engine.
 */
#ifndef FERRITE_FUZZ_ONNX_H
#define FERRITE_FUZZ_ONNX_H

#include <stddef.h>
#include <stdint.h>

/* Feed `size` bytes at `data` to the ONNX parser. Must never crash. */
void fuzz_onnx_input(const uint8_t *data, size_t size);

#endif /* FERRITE_FUZZ_ONNX_H */