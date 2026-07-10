#include "model/tok_map.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

static void test_str_map_put_get(void) {
    str_u32_map m;
    str_u32_map_init(&m, 16);

    str_u32_map_put(&m, "hello", 5, 42);

    uint32_t val = 0;
    bool found = str_u32_map_get(&m, "hello", 5, &val);
    assert(found);
    assert(val == 42);

    str_u32_map_free(&m);
}

static void test_str_map_get_missing(void) {
    str_u32_map m;
    str_u32_map_init(&m, 16);

    str_u32_map_put(&m, "hello", 5, 42);

    uint32_t val = 999;
    bool found = str_u32_map_get(&m, "world", 5, &val);
    assert(!found);
    assert(val == 999);

    str_u32_map_free(&m);
}

static void test_str_map_put_overwrites(void) {
    str_u32_map m;
    str_u32_map_init(&m, 16);

    str_u32_map_put(&m, "key", 3, 10);
    str_u32_map_put(&m, "key", 3, 20);

    uint32_t val = 0;
    bool found = str_u32_map_get(&m, "key", 3, &val);
    assert(found);
    assert(val == 20);

    str_u32_map_free(&m);
}

static void test_str_map_growth(void) {
    str_u32_map m;
    str_u32_map_init(&m, 4);

    char keys[1000][16];
    for (int i = 0; i < 1000; i++) {
        int len = snprintf(keys[i], sizeof(keys[i]), "key_%d", i);
        str_u32_map_put(&m, keys[i], (uint32_t)len, (uint32_t)i);
    }

    for (int i = 0; i < 1000; i++) {
        uint32_t val = 0;
        int len = (int)strlen(keys[i]);
        bool found = str_u32_map_get(&m, keys[i], (uint32_t)len, &val);
        assert(found);
        assert(val == (uint32_t)i);
    }

    str_u32_map_free(&m);
}

static void test_str_map_byte_range_keys(void) {
    str_u32_map m;
    str_u32_map_init(&m, 16);

    const char *buf = "hello";
    str_u32_map_put(&m, buf, 5, 100);
    str_u32_map_put(&m, buf, 4, 200);

    uint32_t val = 0;

    /* "hello" (len 5) */
    assert(str_u32_map_get(&m, buf, 5, &val));
    assert(val == 100);

    /* "hell" (len 4, non-NUL-terminated slice of "hello") */
    assert(str_u32_map_get(&m, buf, 4, &val));
    assert(val == 200);

    /* lookup with a separate buffer whose first 4 bytes match */
    const char other[] = "hellXXX";
    assert(str_u32_map_get(&m, other, 4, &val));
    assert(val == 200);

    /* "hel" (len 3) was never inserted */
    assert(!str_u32_map_get(&m, buf, 3, &val));

    str_u32_map_free(&m);
}

static void test_merge_map_put_get(void) {
    merge_map m;
    merge_map_init(&m, 16);

    merge_map_put(&m, "ab", 2, "c", 1, 10);
    merge_map_put(&m, "a", 1, "bc", 2, 20);

    uint32_t rank = 0;

    /* ("ab","c") -> 10 */
    assert(merge_map_get(&m, "ab", 2, "c", 1, &rank));
    assert(rank == 10);

    /* ("a","bc") -> 20, distinct from ("ab","c") */
    assert(merge_map_get(&m, "a", 1, "bc", 2, &rank));
    assert(rank == 20);

    merge_map_free(&m);
}

static void test_merge_map_miss(void) {
    merge_map m;
    merge_map_init(&m, 16);

    merge_map_put(&m, "ab", 2, "c", 1, 10);

    uint32_t rank = 999;
    assert(!merge_map_get(&m, "x", 1, "y", 1, &rank));
    assert(rank == 999);

    /* right half differs */
    assert(!merge_map_get(&m, "ab", 2, "d", 1, &rank));

    /* left half differs */
    assert(!merge_map_get(&m, "a", 1, "c", 1, &rank));

    merge_map_free(&m);
}

int main(void) {
    test_str_map_put_get();
    test_str_map_get_missing();
    test_str_map_put_overwrites();
    test_str_map_growth();
    test_str_map_byte_range_keys();
    test_merge_map_put_get();
    test_merge_map_miss();
    printf("All tokenizer tests passed.\n");
    return 0;
}
