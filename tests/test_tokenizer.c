#include "model/tok_bpe.h"
#include "model/tok_map.h"
#include "model/tok_unicode.h"
#include "model/tokenizer.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void test_str_map_put_get(void) {
    str_u32_map m;
    assert(str_u32_map_init(&m, 16));

    assert(str_u32_map_put(&m, "hello", 5, 42));

    uint32_t val = 0;
    bool found = str_u32_map_get(&m, "hello", 5, &val);
    assert(found);
    assert(val == 42);

    str_u32_map_free(&m);
}

static void test_str_map_get_missing(void) {
    str_u32_map m;
    assert(str_u32_map_init(&m, 16));

    assert(str_u32_map_put(&m, "hello", 5, 42));

    uint32_t val = 999;
    bool found = str_u32_map_get(&m, "world", 5, &val);
    assert(!found);
    assert(val == 999);

    str_u32_map_free(&m);
}

static void test_str_map_put_overwrites(void) {
    str_u32_map m;
    assert(str_u32_map_init(&m, 16));

    assert(str_u32_map_put(&m, "key", 3, 10));
    assert(str_u32_map_put(&m, "key", 3, 20));

    uint32_t val = 0;
    bool found = str_u32_map_get(&m, "key", 3, &val);
    assert(found);
    assert(val == 20);

    str_u32_map_free(&m);
}

static void test_str_map_growth(void) {
    str_u32_map m;
    assert(str_u32_map_init(&m, 4));

    char keys[1000][16];
    for (int i = 0; i < 1000; i++) {
        int len = snprintf(keys[i], sizeof(keys[i]), "key_%d", i);
        assert(str_u32_map_put(&m, keys[i], (uint32_t)len, (uint32_t)i));
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
    assert(str_u32_map_init(&m, 16));

    const char *buf = "hello";
    assert(str_u32_map_put(&m, buf, 5, 100));
    assert(str_u32_map_put(&m, buf, 4, 200));

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
    assert(merge_map_init(&m, 16));

    assert(merge_map_put(&m, "ab", 2, "c", 1, 10));
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
    assert(merge_map_init(&m, 16));

    const char *s = "abcd";
    assert(merge_map_put(&m, s, 1, s + 1, 3, 0));
    assert(merge_map_put(&m, s, 2, s + 2, 2, 1));
    assert(merge_map_put(&m, s, 3, s + 3, 1, 2));

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
    assert(merge_map_init(&m, 16));

    assert(merge_map_put(&m, "ab", 2, "c", 1, 10));
    assert(merge_map_put(&m, "a", 1, "bc", 2, 20));

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
    assert(merge_map_init(&m, 16));

    assert(merge_map_put(&m, "ab", 2, "c", 1, 10));

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

/* Ids are attacker-controlled sizing input (id_to_token is calloc'd to
 * max_id+1): negative, huge, and non-integer ids must fail the load, matching
 * the Zig reference's behavior. */
static void test_load_rejects_negative_id(void) {
    const char *json = "{\"model\":{\"vocab\":{\"a\":-1},\"merges\":[]}}";
    assert(tokenizer_load_json(json, strlen(json)) == NULL);
}

static void test_load_rejects_huge_id(void) {
    /* 16777216 = 1 << 24, above the 4M id cap (largest real vocab ~262K). */
    const char *json = "{\"model\":{\"vocab\":{\"a\":0,\"b\":16777216},\"merges\":[]}}";
    assert(tokenizer_load_json(json, strlen(json)) == NULL);
}

static void test_load_rejects_non_integer_id(void) {
    const char *json_str = "{\"model\":{\"vocab\":{\"a\":\"x\"},\"merges\":[]}}";
    assert(tokenizer_load_json(json_str, strlen(json_str)) == NULL);

    const char *json_real = "{\"model\":{\"vocab\":{\"a\":1.5},\"merges\":[]}}";
    assert(tokenizer_load_json(json_real, strlen(json_real)) == NULL);
}

/* An unrecognized model.type (e.g. Unigram from T5/ALBERT) must fail the load
 * instead of silently encoding through the SentencePiece-BPE path; absent or
 * "BPE" types keep loading. */
static void test_load_rejects_unigram_model_type(void) {
    const char *unigram = "{\"model\":{\"type\":\"Unigram\",\"vocab\":{\"a\":0}}}";
    assert(tokenizer_load_json(unigram, strlen(unigram)) == NULL);

    const char *bpe = "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":0},\"merges\":[]}}";
    tokenizer_t *tok = tokenizer_load_json(bpe, strlen(bpe));
    assert(tok != NULL);
    tokenizer_free(tok);
}

/* --- bpe_merge tests -------------------------------------------------------- */

/* bpe_node indices are int32_t, so inputs beyond INT32_MAX bytes are
 * unrepresentable: reserve must refuse them without touching the scratch. */
static void test_reserve_rejects_oversized_input(void) {
    encode_scratch s;
    encode_scratch_init(&s);
    assert(encode_scratch_reserve(&s, 8));
    bpe_node *nodes     = s.nodes;
    uint32_t  nodes_cap = s.nodes_cap;

    assert(!encode_scratch_reserve(&s, (size_t)INT32_MAX + 1));
    assert(s.nodes == nodes);
    assert(s.nodes_cap == nodes_cap);

    encode_scratch_free(&s);
}

/* len used to be truncated to uint32_t inside bpe_merge while the loop
 * compared pos < len in size_t: pos stopped advancing and n_nodes overran the
 * scratch. The guard must run before any input byte is read. */
static void test_bpe_merge_rejects_oversized_len(void) {
    const char *json = "{\"model\":{\"vocab\":{\"a\":1,\"b\":2},\"merges\":[]}}";
    tokenizer_t *tok = tokenizer_load_json(json, strlen(json));
    assert(tok != NULL);

    encode_scratch s;
    encode_scratch_init(&s);
    assert(encode_scratch_reserve(&s, 2));

    int32_t *out = NULL;
    assert(bpe_merge(tok, &s, "ab", (size_t)UINT32_MAX + 2, &out) == -1);

    encode_scratch_free(&s);
    tokenizer_free(tok);
}

/* bpe_merge implements BPE/SentencePiece merging only; a WordPiece tokenizer
 * (greedy longest-match, ## continuation) must be refused, not routed through
 * the SentencePiece byte fallback. */
static void test_bpe_merge_rejects_wordpiece(void) {
    const char *json =
        "{\"model\":{\"type\":\"WordPiece\",\"vocab\":{\"a\":1,\"##b\":2,\"[UNK]\":0}}}";
    tokenizer_t *tok = tokenizer_load_json(json, strlen(json));
    assert(tok != NULL);

    encode_scratch s;
    encode_scratch_init(&s);
    assert(encode_scratch_reserve(&s, 2));

    int32_t *out = NULL;
    assert(bpe_merge(tok, &s, "ab", 2, &out) == -1);

    encode_scratch_free(&s);
    tokenizer_free(tok);
}

static void test_bpe_abcd(void) {
    const char *json =
        "{\"model\":{\"vocab\":{\"a\":1,\"b\":2,\"c\":3,\"d\":4,\"ab\":5,\"cd\":6,\"abcd\":7},"
        "\"merges\":[[\"c\",\"d\"],[\"a\",\"b\"],[\"ab\",\"cd\"]]}}";
    tokenizer_t *tok = tokenizer_load_json(json, strlen(json));
    assert(tok != NULL);

    encode_scratch s;
    encode_scratch_init(&s);
    assert(encode_scratch_reserve(&s, 4));

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
    assert(encode_scratch_reserve(&s, 3));

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
    assert(encode_scratch_reserve(&s, 5));

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
    assert(encode_scratch_reserve(&s, 10));

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
    assert(encode_scratch_reserve(&s, 3));

    int32_t *out   = NULL;
    int      count = bpe_merge(tok, &s, "abc", 3, &out);
    assert(count == 2);
    assert(out[0] == 1); /* "a" */
    assert(out[1] == 4); /* "bc" */

    encode_scratch_free(&s);
    tokenizer_free(tok);
}

/* Byte-level BPE with printable ASCII: both symbols are direct vocab hits, so
 * this never enters the fallback (GPT-2 byte-to-unicode maps printables to
 * themselves) - characterization of the direct-hit path under ByteLevel. */
static void test_bpe_bytelevel_az(void) {
    const char *json =
        "{\"pre_tokenizer\":{\"type\":\"ByteLevel\"},"
        "\"model\":{\"vocab\":{\"a\":1,\"z\":99},\"merges\":[]}}";
    tokenizer_t *tok = tokenizer_load_json(json, strlen(json));
    assert(tok != NULL);

    encode_scratch s;
    encode_scratch_init(&s);
    assert(encode_scratch_reserve(&s, 2));

    int32_t *out   = NULL;
    int      count = bpe_merge(tok, &s, "az", 2, &out);
    assert(count == 2);
    assert(out[0] == 1);
    assert(out[1] == 99);

    encode_scratch_free(&s);
    tokenizer_free(tok);
}

/* Encode a codepoint <= 0x7FF as UTF-8 (enough for the byte table's max 323). */
static int test_utf8_encode(uint32_t cp, char *buf) {
    if (cp < 0x80) {
        buf[0] = (char)cp;
        return 1;
    }
    assert(cp < 0x800);
    buf[0] = (char)(0xC0 | (cp >> 6));
    buf[1] = (char)(0x80 | (cp & 0x3F));
    return 2;
}

/* Byte-level fallback: input "\xC3\xA9" is ONE UTF-8 char ("é"), so it forms
 * a single symbol that misses the vocab. The fallback must emit one token per
 * RAW byte, keyed by the UTF-8 form of that byte's GPT-2 unicode mapping -
 * which differs from the raw byte string for any byte >= 0x80. */
static void test_bpe_bytelevel_fallback(void) {
    uc_bytes_unicode_t table;
    uc_build_bytes_to_unicode(&table);

    char key_c3[5] = {0};
    char key_a9[5] = {0};
    test_utf8_encode(table.byte_to_cp[0xC3], key_c3);
    test_utf8_encode(table.byte_to_cp[0xA9], key_a9);

    char json[256];
    int  n = snprintf(json, sizeof(json),
                      "{\"pre_tokenizer\":{\"type\":\"ByteLevel\"},"
                      "\"model\":{\"vocab\":{\"%s\":50,\"%s\":51},\"merges\":[]}}",
                      key_c3, key_a9);
    assert(n > 0 && (size_t)n < sizeof(json));

    tokenizer_t *tok = tokenizer_load_json(json, (size_t)n);
    assert(tok != NULL);

    encode_scratch s;
    encode_scratch_init(&s);
    assert(encode_scratch_reserve(&s, 2));

    int32_t *out   = NULL;
    int      count = bpe_merge(tok, &s, "\xC3\xA9", 2, &out);
    assert(count == 2);
    assert(out[0] == 50);
    assert(out[1] == 51);

    encode_scratch_free(&s);
    tokenizer_free(tok);
}

/* Byte-level BPE whose vocab is missing a mapped byte token (pathological,
 * e.g. truncated vocab): the byte must fall back to <unk> when present rather
 * than being silently dropped; with no <unk> either, it is skipped. */
static void test_bpe_bytelevel_fallback_unk_on_vocab_miss(void) {
    const char *json =
        "{\"pre_tokenizer\":{\"type\":\"ByteLevel\"},"
        "\"model\":{\"vocab\":{\"<unk>\":7},\"merges\":[]}}";
    tokenizer_t *tok = tokenizer_load_json(json, strlen(json));
    assert(tok != NULL);

    encode_scratch s;
    encode_scratch_init(&s);
    assert(encode_scratch_reserve(&s, 1));

    int32_t *out   = NULL;
    int      count = bpe_merge(tok, &s, "a", 1, &out);
    assert(count == 1);
    assert(out[0] == 7);

    tokenizer_free(tok);

    /* Neither the byte token nor <unk>: the byte is skipped. */
    const char *json_no_unk =
        "{\"pre_tokenizer\":{\"type\":\"ByteLevel\"},"
        "\"model\":{\"vocab\":{\"z\":1},\"merges\":[]}}";
    tok = tokenizer_load_json(json_no_unk, strlen(json_no_unk));
    assert(tok != NULL);

    assert(bpe_merge(tok, &s, "a", 1, &out) == 0);

    encode_scratch_free(&s);
    tokenizer_free(tok);
}

/* SentencePiece byte fallback: an unknown byte becomes its <0xNN> token. */
static void test_bpe_sp_hexbyte(void) {
    const char *json = "{\"model\":{\"vocab\":{\"a\":12,\"<0x0A>\":10},\"merges\":[]}}";
    tokenizer_t *tok = tokenizer_load_json(json, strlen(json));
    assert(tok != NULL);

    encode_scratch s;
    encode_scratch_init(&s);
    assert(encode_scratch_reserve(&s, 2));

    int32_t *out   = NULL;
    int      count = bpe_merge(tok, &s, "a\n", 2, &out);
    assert(count == 2);
    assert(out[0] == 12);
    assert(out[1] == 10);

    encode_scratch_free(&s);
    tokenizer_free(tok);
}

/* SentencePiece without byte entries: unknown bytes fall back to <unk>. */
static void test_bpe_sp_unk(void) {
    const char *json = "{\"model\":{\"vocab\":{\"a\":12,\"<unk>\":0},\"merges\":[]}}";
    tokenizer_t *tok = tokenizer_load_json(json, strlen(json));
    assert(tok != NULL);

    encode_scratch s;
    encode_scratch_init(&s);
    assert(encode_scratch_reserve(&s, 2));

    int32_t *out   = NULL;
    int      count = bpe_merge(tok, &s, "a\n", 2, &out);
    assert(count == 2);
    assert(out[0] == 12);
    assert(out[1] == 0);

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
    test_load_rejects_negative_id();
    test_load_rejects_huge_id();
    test_load_rejects_non_integer_id();
    test_load_rejects_unigram_model_type();
    test_reserve_rejects_oversized_input();
    test_bpe_merge_rejects_oversized_len();
    test_bpe_merge_rejects_wordpiece();
    test_bpe_abcd();
    test_bpe_aaa();
    test_bpe_aaaaa();
    test_bpe_hellohello();
    test_bpe_version_restale();
    test_bpe_bytelevel_az();
    test_bpe_bytelevel_fallback();
    test_bpe_bytelevel_fallback_unk_on_vocab_miss();
    test_bpe_sp_hexbyte();
    test_bpe_sp_unk();
    printf("All tokenizer tests passed.\n");
    return 0;
}
