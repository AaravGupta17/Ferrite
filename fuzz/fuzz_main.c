/*
 * fuzz/fuzz_main.c — standalone driver for the ONNX fuzz target.
 *
 * This is the non-libFuzzer entry point (CTest `fuzz_runner`): every path
 * named on the command line is fed to fuzz_onnx_input() — a corpus replay
 * that exercises the identical code path libFuzzer uses, so toolchains
 * without clang's fuzzer+trace-pc-guard instrumentation still get a runnable
 * fuzz check. Exit code is 0 as long as nothing crashed (the parser either
 * loads or fails loudly by contract).
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "fuzz_onnx.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <corpus file|dir entries>...\n", argv[0]);
        return 2;
    }

    for (int i = 1; i < argc; i++) {
        FILE *f = fopen(argv[i], "rb");
        if (!f) {
            fprintf(stderr, "fuzz: cannot open %s\n", argv[i]);
            return 2;
        }
        fseek(f, 0, SEEK_END);
        long len = ftell(f);
        rewind(f);
        if (len > 0 && len < 16 * 1024 * 1024) {
            uint8_t *buf = (uint8_t *)malloc((size_t)len);
            if (buf) {
                if (fread(buf, 1, (size_t)len, f) == (size_t)len)
                    fuzz_onnx_input(buf, (size_t)len);
                free(buf);
            }
        }
        fclose(f);
    }

    printf("fuzz: replayed %d corpus files, no crash\n", argc - 1);
    return 0;
}