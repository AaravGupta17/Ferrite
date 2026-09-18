/*
 * fuzz/fuzz_onnx_libfuzzer.c — libFuzzer entry shim (Section 1.3).
 *
 * libFuzzer's runtime provides main() and calls LLVMFuzzerTestOneInput. The
 * shared parser target lives in fuzz_onnx.c as fuzz_onnx_input; this TU is
 * compiled into fuzz_onnx_fuzzer only, so the standalone corpus-replay path
 * (fuzz_runner + fuzz_main.c) never sees it.
 */
#include "fuzz_onnx.h"

int LLVMFuzzerTestOneInput(const unsigned char *data, size_t size) {
    fuzz_onnx_input(data, size);
    return 0;
}