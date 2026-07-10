#include "model/tok_bpe.h"
#include "model/tok_map.h"
#include "model/tokenizer.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
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

/* --- merge_hash exposing tests -------------------------------------------- */

#define TEST_FNV_INIT  0xcbf29ce484222325ULL
#define TEST_FNV_PRIME 0x100000001b3ULL

static uint64_t test_fnv1a_update(uint64_t h, const char *data, uint32_t len) {
    for (uint32_t i = 0; i < len; i++) {
        h ^= (uint8_t)data[i];
        h *= TEST_FNV_PRIME;
    }
    return h;
}

static uint64_t test_fnv1a(const char *data, uint32_t len) {
    return test_fnv1a_update(TEST_FNV_INIT, data, len);
}

static uint64_t find_merge_hash(const merge_map *m, const char *l, uint32_t llen,
                                const char *r, uint32_t rlen) {
    for (uint32_t i = 0; i < m->cap; i++) {
        const merge_entry *e = &m->entries[i];
        if (e->l && e->llen == llen && e->rlen == rlen &&
            memcmp(e->l, l, llen) == 0 && memcmp(e->r, r, rlen) == 0)
            return e->hash;
    }
    assert(0 && "merge entry not found");
    return 0;
}

static void test_merge_hash_separator_not_noop(void) {
    merge_map m;
    merge_map_init(&m, 16);

    merge_map_put(&m, "ab", 2, "c", 1, 10);
    uint64_t actual = find_merge_hash(&m, "ab", 2, "c", 1);

    /* What a no-op separator (h ^= 0x00; h *= PRIME) would produce */
    uint64_t h_noop = test_fnv1a("ab", 2);
    h_noop *= TEST_FNV_PRIME;
    h_noop = test_fnv1a_update(h_noop, "c", 1);

    assert(actual != h_noop);

    merge_map_free(&m);
}

static void test_merge_hash_split_positions_distinct(void) {
    merge_map m;
    merge_map_init(&m, 16);

    const char *s = "abcd";
    merge_map_put(&m, s, 1, s + 1, 3, 0);
    merge_map_put(&m, s, 2, s + 2, 2, 1);
    merge_map_put(&m, s, 3, s + 3, 1, 2);

    uint64_t h1 = find_merge_hash(&m, s, 1, s + 1, 3);
    uint64_t h2 = find_merge_hash(&m, s, 2, s + 2, 2);
    uint64_t h3 = find_merge_hash(&m, s, 3, s + 3, 1);

    assert(h1 != h2);
    assert(h1 != h3);
    assert(h2 != h3);

    /* None should equal plain fnv1a("abcd") - separator must prevent this */
    uint64_t plain = test_fnv1a("abcd", 4);
    assert(h1 != plain);
    assert(h2 != plain);
    assert(h3 != plain);

    merge_map_free(&m);
}

/* --- merge_map tests ------------------------------------------------------ */

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

/* --- tokenizer_load_json tests --------------------------------------------- */

static void test_load_json_minimal(void) {
    const char *json = "{\"model\":{\"vocab\":{\"a\":1,\"b\":2},\"merges\":[]}}";
    tokenizer_t *tok = tokenizer_load_json(json, strlen(json));
    assert(tok != NULL);
    assert(tokenizer_vocab_size(tok) == 2);
    assert(strcmp(tokenizer_decode_token(tok, 1), "a") == 0);
    assert(strcmp(tokenizer_decode_token(tok, 2), "b") == 0);
    assert(tokenizer_decode_token(tok, 999) == NULL);
    tokenizer_free(tok);
}

static void test_load_json_gpt2_smoke(void) {
    const char *path = MLXD_FIXTURES_DIR "/gpt2/tokenizer.json";
    FILE       *f    = fopen(path, "rb");
    assert(f != NULL);
    assert(fseek(f, 0, SEEK_END) == 0);
    long sz = ftell(f);
    assert(sz > 0);
    assert(fseek(f, 0, SEEK_SET) == 0);
    char *buf = malloc((size_t)sz);
    assert(buf != NULL);
    assert(fread(buf, 1, (size_t)sz, f) == (size_t)sz);
    fclose(f);

    tokenizer_t *tok = tokenizer_load_json(buf, (size_t)sz);
    assert(tok != NULL);
    assert(tokenizer_vocab_size(tok) == 50257);
    tokenizer_free(tok);
    free(buf);
}

/* --- bpe_merge tests -------------------------------------------------------- */

static void test_bpe_abcd(void) {
    const char *json =
        "{\"model\":{\"vocab\":{\"a\":1,\"b\":2,\"c\":3,\"d\":4,\"ab\":5,\"cd\":6,\"abcd\":7},"
        "\"merges\":[[\"c\",\"d\"],[\"a\",\"b\"],[\"ab\",\"cd\"]]}}";
    tokenizer_t *tok = tokenizer_load_json(json, strlen(json));
    assert(tok != NULL);

    encode_scratch s;
    encode_scratch_init(&s);
    encode_scratch_reserve(&s, 4);

    int32_t *out   = NULL;
    int      count = bpe_merge(tok, &s, "abcd", 4, &out);
    assert(count == 1);
    assert(out[0] == 7);

    encode_scratch_free(&s);
    tokenizer_free(tok);
}

static void test_bpe_aaa(void) {
    const char *json = "{\"model\":{\"vocab\":{\"a\":1,\"aa\":2},\"merges\":[[\"a\",\"a\"]]}}";
    tokenizer_t *tok = tokenizer_load_json(json, strlen(json));
    assert(tok != NULL);

    encode_scratch s;
    encode_scratch_init(&s);
    encode_scratch_reserve(&s, 3);

    int32_t *out   = NULL;
    int      count = bpe_merge(tok, &s, "aaa", 3, &out);
    assert(count == 2);
    assert(out[0] == 2); /* leftmost pair merges first: "aa" */
    assert(out[1] == 1); /* trailing "a" survives */

    encode_scratch_free(&s);
    tokenizer_free(tok);
}

/* Five 'a's force >= 3 tied candidates in the heap at once: a heap without a
 * leftmost tie-break pops (3,4) before (2,3) after the first merge, and the
 * resulting stale candidates merge through unlinked nodes into "aaa" (a vocab
 * miss). Leftmost-wins order gives [aa, aa, a]. */
static void test_bpe_aaaaa(void) {
    const char *json = "{\"model\":{\"vocab\":{\"a\":1,\"aa\":2},\"merges\":[[\"a\",\"a\"]]}}";
    tokenizer_t *tok = tokenizer_load_json(json, strlen(json));
    assert(tok != NULL);

    encode_scratch s;
    encode_scratch_init(&s);
    encode_scratch_reserve(&s, 5);

    int32_t *out   = NULL;
    int      count = bpe_merge(tok, &s, "aaaaa", 5, &out);
    assert(count == 3);
    assert(out[0] == 2);
    assert(out[1] == 2);
    assert(out[2] == 1);

    encode_scratch_free(&s);
    tokenizer_free(tok);
}

/* Cascading merges: he+ll, hell+o build up through intermediate symbols. */
static void test_bpe_hellohello(void) {
    const char *json =
        "{\"model\":{\"vocab\":{\"h\":1,\"e\":2,\"l\":3,\"o\":4,\"he\":5,\"ll\":6,\"hell\":7,"
        "\"hello\":8},"
        "\"merges\":[[\"h\",\"e\"],[\"l\",\"l\"],[\"he\",\"ll\"],[\"hell\",\"o\"]]}}";
    tokenizer_t *tok = tokenizer_load_json(json, strlen(json));
    assert(tok != NULL);

    encode_scratch s;
    encode_scratch_init(&s);
    encode_scratch_reserve(&s, 10);

    int32_t *out   = NULL;
    int      count = bpe_merge(tok, &s, "hellohello", 10, &out);
    assert(count == 2);
    assert(out[0] == 8);
    assert(out[1] == 8);

    encode_scratch_free(&s);
    tokenizer_free(tok);
}

/* Re-stale through a live node: after (b,c) merges, node 1 is still linked
 * and adjacent to node 0, but its content grew from "b" to "bc" - the seeded
 * (a,b) candidate must be rejected. Only version stamps catch this; the
 * adjacency guard alone wrongly merges to "abc" = [5]. */
static void test_bpe_version_restale(void) {
    const char *json =
        "{\"model\":{\"vocab\":{\"a\":1,\"b\":2,\"c\":3,\"bc\":4,\"abc\":5},"
        "\"merges\":[[\"b\",\"c\"],[\"a\",\"b\"]]}}";
    tokenizer_t *tok = tokenizer_load_json(json, strlen(json));
    assert(tok != NULL);

    encode_scratch s;
    encode_scratch_init(&s);
    encode_scratch_reserve(&s, 3);

    int32_t *out   = NULL;
    int      count = bpe_merge(tok, &s, "abc", 3, &out);
    assert(count == 2);
    assert(out[0] == 1); /* "a" */
    assert(out[1] == 4); /* "bc" */

    encode_scratch_free(&s);
    tokenizer_free(tok);
}

int main(void) {
    test_str_map_put_get();
    test_str_map_get_missing();
    test_str_map_put_overwrites();
    test_str_map_growth();
    test_str_map_byte_range_keys();
    test_merge_hash_separator_not_noop();
    test_merge_hash_split_positions_distinct();
    test_merge_map_put_get();
    test_merge_map_miss();
    test_load_json_minimal();
    test_load_json_gpt2_smoke();
    test_bpe_abcd();
    test_bpe_aaa();
    test_bpe_aaaaa();
    test_bpe_hellohello();
    test_bpe_version_restale();
    printf("All tokenizer tests passed.\n");
    return 0;
}
