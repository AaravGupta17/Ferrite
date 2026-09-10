/*
 * fuzz/fuzz_onnx.c — the ONNX-parser fuzz target (Section 1.3).
 *
 * The public contract of every Ferrite importer path is "fail loudly, never
 * crash": fe_onnx_load returns a FeStatus error for malformed input. This
 * harness exercises exactly that boundary with arbitrary bytes. Under
 * libFuzzer + ASan/UBSan (clang -fsanitize=fuzzer,address,undefined) it is
 * the continuous-fuzzing layer; under the standalone main (see fuzz_main.c)
 * it replays a fixed corpus. See fuzz_onnx.h for the split.
 */
#include "fuzz_onnx.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../core/allocator.h"
#include "../graph/graph.h"
#include "../importer/onnx.h"

#define FUZZ_TMP_PATH "fuzz/onnx_fuzz_input.bin"
#define FUZZ_WEIGHT_ARENA_BYTES (4u * 1024u * 1024u)

void fuzz_onnx_input(const uint8_t *data, size_t size) {
    /* Write the fuzz bytes to a temp file exactly as received. */
    FILE *f = fopen(FUZZ_TMP_PATH, "wb");
    if (!f) return;
    size_t written = fwrite(data, 1, size, f);
    fclose(f);
    if (written != size) return;

    FeGraph g;
    memset(&g, 0, sizeof(g));

    unsigned char *arena_buf =
        (unsigned char *)malloc(FUZZ_WEIGHT_ARENA_BYTES);
    if (!arena_buf) { remove(FUZZ_TMP_PATH); return; }
    FeArena wa;
    if (fe_arena_init(&wa, arena_buf, FUZZ_WEIGHT_ARENA_BYTES) == FE_OK) {
        /* Ignore the status: we only care that it returns (never crashes),
         * and that on success the graph is well-formed. */
        FeStatus s = fe_onnx_load(&g, &wa, FUZZ_TMP_PATH);
        (void)s;
    }

    free(arena_buf);
    remove(FUZZ_TMP_PATH);
}