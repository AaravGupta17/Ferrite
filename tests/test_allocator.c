// tests/test_allocator.c
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include "../core/allocator.h"

#define ARENA_SIZE (1024 * 1024)  /* 1MB test arena */
static unsigned char arena_buf[ARENA_SIZE] __attribute__((aligned(64)));


static void test_basic_alloc(void) {
    FeArena a;
    fe_arena_init(&a, arena_buf, ARENA_SIZE);

    void *p1 = fe_arena_alloc(&a, 64, 8);
    void *p2 = fe_arena_alloc(&a, 128, 8);

    assert(p1 != NULL);
    assert(p2 != NULL);
    assert((uintptr_t)p1 % 8 == 0);   /* alignment respected */
    assert((uintptr_t)p2 % 8 == 0);
    assert((unsigned char *)p2 >= (unsigned char *)p1 + 64);

    printf("PASS test_basic_alloc\n");
}

static void test_alignment(void) {
    FeArena a;
    fe_arena_init(&a, arena_buf, ARENA_SIZE);

    /* Force a misaligned starting offset */
    fe_arena_alloc(&a, 3, 1);

    void *p = fe_arena_alloc(&a, 16, 64);
    assert(p != NULL);
    assert((uintptr_t)p % 64 == 0);   /* must be 64-byte aligned */

    printf("PASS test_alignment\n");
}

static void test_reset(void) {
    FeArena a;
    fe_arena_init(&a, arena_buf, ARENA_SIZE);

    void *p1 = fe_arena_alloc(&a, 256, 8);
    assert(fe_arena_used(&a) > 0);

    fe_arena_reset(&a);
    assert(fe_arena_used(&a) == 0);

    /* After reset, same address should be returned */
    void *p2 = fe_arena_alloc(&a, 256, 8);
    assert(p1 == p2);

    printf("PASS test_reset\n");
}

static void test_exhaustion(void) {
    FeArena a;
    unsigned char small_buf[128];
    fe_arena_init(&a, small_buf, 128);

    void *p1 = fe_arena_alloc(&a, 64, 8);
    void *p2 = fe_arena_alloc(&a, 64, 8);
    void *p3 = fe_arena_alloc(&a, 8, 8);   /* should fail */

    assert(p1 != NULL);
    assert(p2 != NULL);
    assert(p3 == NULL);

    printf("PASS test_exhaustion\n");
}

static void test_arena_tensor(void) {
    FeArena a;
    fe_arena_init(&a, arena_buf, ARENA_SIZE);

    int shape[] = {3, 4};
    FeTensor *t = fe_arena_alloc_tensor(&a, DTYPE_FLOAT32, 2, shape);

    assert(t != NULL);
    assert(t->owns_data == false);
    assert(t->strides[0] == 4);
    assert(t->strides[1] == 1);
    assert(t->nbytes == 3 * 4 * 4);

    /* Write and read back */
    int idx[] = {1, 2};
    fe_tensor_set_f32(t, idx, 99.0f);
    assert(fe_tensor_get_f32(t, idx) == 99.0f);

    size_t used = fe_arena_used(&a);
    fe_arena_reset(&a);
    assert(fe_arena_used(&a) == 0);
    assert(fe_arena_peak(&a) == used);  /* peak preserved */

    printf("PASS test_arena_tensor\n");
}

int main(void) {
    test_basic_alloc();
    test_alignment();
    test_reset();
    test_exhaustion();
    test_arena_tensor();
    printf("\nAll tests passed.\n");
    return 0;
}