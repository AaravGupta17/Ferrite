// tests/test_tensor_ser.c
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "../core/tensor.h"
#include "../core/tensor_ser.h"

#define TMP_A "test_ser_tmp_a.bin"
#define TMP_B "test_ser_tmp_b.bin"
#define TMP_C "test_ser_tmp_c.bin"

static void fill_f32(FeTensor *t, float base) {
    float *p = (float *)t->data;
    for (size_t i = 0; i < t->nbytes / sizeof(float); i++) p[i] = base + (float)i;
}

static void assert_same_bytes(const FeTensor *a, const FeTensor *b) {
    assert(a->dtype == b->dtype);
    assert(a->ndim == b->ndim);
    for (int d = 0; d < a->ndim; d++) assert(a->shape[d] == b->shape[d]);
    assert(a->nbytes == b->nbytes);
    assert(memcmp(a->data, b->data, a->nbytes) == 0);
}

static void test_roundtrip_f32(void) {
    int shape[] = {3, 4};
    FeTensor *t = fe_tensor_alloc(DTYPE_FLOAT32, 2, shape);
    fill_f32(t, 0.5f);

    assert(fe_tensor_save(t, TMP_A) == FE_OK);

    FeTensor *loaded = NULL;
    assert(fe_tensor_load(TMP_A, &loaded) == FE_OK);
    assert(loaded != NULL && loaded->owns_data == true);
    assert_same_bytes(t, loaded);

    fe_tensor_free(loaded);
    fe_tensor_free(t);
    remove(TMP_A);
    printf("PASS test_roundtrip_f32\n");
}

static void test_roundtrip_int_dtypes(void) {
    int shape[] = {6};
    FeTensor *ti8  = fe_tensor_alloc(DTYPE_INT8,  1, shape);
    FeTensor *ti32 = fe_tensor_alloc(DTYPE_INT32, 1, shape);
    FeTensor *tf64 = fe_tensor_alloc(DTYPE_FLOAT64, 1, shape);
    for (int i = 0; i < 6; i++) {
        ((int8_t *)ti8->data)[i]  = (int8_t)(i - 3);
        ((int32_t *)ti32->data)[i] = i * 1000;
        ((double *)tf64->data)[i] = i * 0.25;
    }

    assert(fe_tensor_save(ti8,  TMP_A) == FE_OK);
    assert(fe_tensor_save(ti32, TMP_B) == FE_OK);
    assert(fe_tensor_save(tf64, TMP_C) == FE_OK);

    FeTensor *li8 = NULL, *li32 = NULL, *lf64 = NULL;
    assert(fe_tensor_load(TMP_A, &li8)  == FE_OK);
    assert(fe_tensor_load(TMP_B, &li32) == FE_OK);
    assert(fe_tensor_load(TMP_C, &lf64) == FE_OK);
    assert(li8->dtype  == DTYPE_INT8);
    assert(li32->dtype == DTYPE_INT32);
    assert(lf64->dtype == DTYPE_FLOAT64);
    assert_same_bytes(ti8, li8);
    assert_same_bytes(ti32, li32);
    assert_same_bytes(tf64, lf64);

    fe_tensor_free(li8);
    fe_tensor_free(li32);
    fe_tensor_free(lf64);
    fe_tensor_free(ti8);
    fe_tensor_free(ti32);
    fe_tensor_free(tf64);
    remove(TMP_A);
    remove(TMP_B);
    remove(TMP_C);
    printf("PASS test_roundtrip_int_dtypes\n");
}

static void test_roundtrip_max_dims(void) {
    int shape[FERRITE_MAX_DIMS] = {2, 1, 2, 1, 2, 1, 2, 3};   /* numel = 48 */
    FeTensor *t = fe_tensor_alloc(DTYPE_FLOAT32, FERRITE_MAX_DIMS, shape);
    fill_f32(t, -2.25f);

    assert(fe_tensor_save(t, TMP_A) == FE_OK);

    FeTensor *loaded = NULL;
    assert(fe_tensor_load(TMP_A, &loaded) == FE_OK);
    assert_same_bytes(t, loaded);

    fe_tensor_free(loaded);
    fe_tensor_free(t);
    remove(TMP_A);
    printf("PASS test_roundtrip_max_dims\n");
}

static void test_save_view_repacks(void) {
    /* A transposed view is non-contiguous; save must store its logical
     * contents in canonical row-major form. */
    int shape[] = {3, 4};
    FeTensor *t  = fe_tensor_alloc(DTYPE_FLOAT32, 2, shape);
    fill_f32(t, 1.0f);
    FeTensor *tr = fe_tensor_transpose(t, 0, 1);   /* 4x3 view */

    assert(fe_tensor_save(tr, TMP_A) == FE_OK);

    FeTensor *loaded = NULL;
    assert(fe_tensor_load(TMP_A, &loaded) == FE_OK);
    assert(loaded->shape[0] == 4 && loaded->shape[1] == 3);
    assert(loaded->owns_data == true);

    /* Loaded bytes must equal the canonical repack of the view. */
    FeTensor *canonical = fe_tensor_contiguous(tr);
    assert(memcmp(loaded->data, canonical->data, loaded->nbytes) == 0);

    fe_tensor_free(canonical);
    fe_tensor_free(loaded);
    fe_tensor_free(tr);
    fe_tensor_free(t);
    remove(TMP_A);
    printf("PASS test_save_view_repacks\n");
}

/* Write a minimal VALID v1 file (2x2 float32) for corruption tests,
 * then hand-mutate specific fields. */
static size_t write_valid_minimal(const char *path) {
    FILE *f = fopen(path, "wb");
    assert(f != NULL);
    const unsigned char hdr[] = {
        'F','E','T','N',                    /* magic          */
        0x01,0x00,0x00,0x00,                /* version = 1    */
        0x00,0x00,0x00,0x00,                /* dtype FLOAT32  */
        0x02,0x00,0x00,0x00,                /* ndim = 2       */
        0x02,0x00,0x00,0x00,                /* shape[0] = 2   */
        0x02,0x00,0x00,0x00,                /* shape[1] = 2   */
        0x10,0x00,0x00,0x00,0x00,0x00,0x00,0x00, /* data_len = 16 */
        0,1,2,3, 4,5,6,7, 8,9,10,11, 12,13,14,15, /* payload     */
    };
    fwrite(hdr, 1, sizeof(hdr), f);
    fclose(f);
    return sizeof(hdr);
}

/* Copy `path`, dropping the last byte (truncated payload). */
static void write_truncated_copy(const char *src, const char *dst, size_t len) {
    unsigned char buf[64];
    FILE *f = fopen(src, "rb");
    assert(f != NULL && fread(buf, 1, len, f) == len);
    fclose(f);
    f = fopen(dst, "wb");
    assert(f != NULL);
    fwrite(buf, 1, len - 1, f);
    fclose(f);
}

static void test_corruption_rejected(void) {
    size_t len = write_valid_minimal(TMP_A);

    FeTensor *out = NULL;

    /* Golden load passes first. */
    assert(fe_tensor_load(TMP_A, &out) == FE_OK);
    fe_tensor_free(out);
    out = NULL;

    /* Mutate one byte of a fresh golden file per case, so every
     * rejection is attributable to exactly that field. */
#define MUTATE(off, b)                            \
    do {                                          \
        write_valid_minimal(TMP_A);               \
        FILE *mf = fopen(TMP_A, "r+b");           \
        assert(mf != NULL);                       \
        fseek(mf, (off), SEEK_SET);               \
        fputc((b), mf);                           \
        fclose(mf);                               \
    } while (0)

    /* Bad magic. */
    MUTATE(0, 'X');
    assert(fe_tensor_load(TMP_A, &out) == FE_ERR_IO);

    /* Unsupported future version. */
    MUTATE(4, 99);                       /* version low byte -> 99 */
    assert(fe_tensor_load(TMP_A, &out) == FE_ERR_IO);

    /* Unknown dtype (77 is not a FeDtype). */
    MUTATE(8, 77);
    assert(fe_tensor_load(TMP_A, &out) == FE_ERR_DTYPE);

    /* Impossible rank: 0 and > FERRITE_MAX_DIMS. */
    MUTATE(12, 0);
    assert(fe_tensor_load(TMP_A, &out) == FE_ERR_SHAPE);

    MUTATE(12, FERRITE_MAX_DIMS + 1);
    assert(fe_tensor_load(TMP_A, &out) == FE_ERR_SHAPE);

    /* Zero-length axis. */
    MUTATE(20, 0);                       /* shape[1] low byte */
    assert(fe_tensor_load(TMP_A, &out) == FE_ERR_SHAPE);

    /* data_len inconsistent with computed nbytes: 16 -> 255. */
    MUTATE(24, 0xFF);                    /* data_len low byte */
    assert(fe_tensor_load(TMP_A, &out) == FE_ERR_IO);

#undef MUTATE

    /* Truncated payload. */
    write_valid_minimal(TMP_A);
    write_truncated_copy(TMP_A, TMP_B, len);
    assert(fe_tensor_load(TMP_B, &out) == FE_ERR_IO);

    /* Trailing garbage violates strict v1. */
    FILE *f = fopen(TMP_A, "ab");
    fputc('Z', f);
    fclose(f);
    assert(fe_tensor_load(TMP_A, &out) == FE_ERR_IO);

    /* Missing file. */
    assert(fe_tensor_load("test_ser_no_such_file.bin", &out) == FE_ERR_IO);

    /* NULL arguments. */
    FeTensor *t = fe_tensor_alloc(DTYPE_FLOAT32, 1, (int[]){2});
    assert(fe_tensor_save(NULL, TMP_A) == FE_ERR_NULL);
    assert(fe_tensor_save(t, NULL) == FE_ERR_NULL);
    assert(fe_tensor_load(NULL, &out) == FE_ERR_NULL);
    assert(fe_tensor_load(TMP_A, NULL) == FE_ERR_NULL);
    fe_tensor_free(t);

    remove(TMP_A);
    remove(TMP_B);
    printf("PASS test_corruption_rejected\n");
}

int main(void) {
    test_roundtrip_f32();
    test_roundtrip_int_dtypes();
    test_roundtrip_max_dims();
    test_save_view_repacks();
    test_corruption_rejected();
    printf("\nAll serialization tests passed.\n");
    return 0;
}
