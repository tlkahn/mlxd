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
    const char *json = "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":1,\"b\":2},\"merges\":[]}}";
    tokenizer_t *tok = tokenizer_load_json(json, strlen(json));
    assert(tok != NULL);
    assert(tokenizer_vocab_size(tok) == 2);
    assert(strcmp(tokenizer_decode_token(tok, 1), "a") == 0);
    assert(strcmp(tokenizer_decode_token(tok, 2), "b") == 0);
    assert(tokenizer_decode_token(tok, 999) == NULL);
    tokenizer_free(tok);
}

/* Characterization for the id_to_token fill: sparse, unordered ids must all
 * decode back to their keys and every gap id must decode NULL, regardless of
 * how the loader walks the vocab. */
static void test_load_json_sparse_unordered_ids(void) {
    const char *json = "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"x\":7,\"y\":0,\"z\":3},\"merges\":[]}}";
    tokenizer_t *tok = tokenizer_load_json(json, strlen(json));
    assert(tok != NULL);
    assert(tokenizer_vocab_size(tok) == 3);
    assert(strcmp(tokenizer_decode_token(tok, 7), "x") == 0);
    assert(strcmp(tokenizer_decode_token(tok, 0), "y") == 0);
    assert(strcmp(tokenizer_decode_token(tok, 3), "z") == 0);
    assert(tokenizer_decode_token(tok, 1) == NULL);
    assert(tokenizer_decode_token(tok, 2) == NULL);
    assert(tokenizer_decode_token(tok, 4) == NULL);
    assert(tokenizer_decode_token(tok, 5) == NULL);
    assert(tokenizer_decode_token(tok, 6) == NULL);
    assert(tokenizer_decode_token(tok, 8) == NULL);
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
    const char *json = "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":-1},\"merges\":[]}}";
    assert(tokenizer_load_json(json, strlen(json)) == NULL);
}

static void test_load_rejects_huge_id(void) {
    /* 16777216 = 1 << 24, above the 4M id cap (largest real vocab ~262K). */
    const char *json = "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":0,\"b\":16777216},\"merges\":[]}}";
    assert(tokenizer_load_json(json, strlen(json)) == NULL);
}

static void test_load_rejects_non_integer_id(void) {
    const char *json_str = "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":\"x\"},\"merges\":[]}}";
    assert(tokenizer_load_json(json_str, strlen(json_str)) == NULL);

    const char *json_real = "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":1.5},\"merges\":[]}}";
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

/* A present-but-non-string model.type is malformed input: it must fail the
 * load rather than fall through the string checks into BPE/SP detection. */
static void test_load_rejects_non_string_model_type(void) {
    const char *num = "{\"model\":{\"type\":42,\"vocab\":{\"a\":0},\"merges\":[]}}";
    assert(tokenizer_load_json(num, strlen(num)) == NULL);

    const char *boolean = "{\"model\":{\"type\":true,\"vocab\":{\"a\":0},\"merges\":[]}}";
    assert(tokenizer_load_json(boolean, strlen(boolean)) == NULL);

    const char *arr = "{\"model\":{\"type\":[],\"vocab\":{\"a\":0},\"merges\":[]}}";
    assert(tokenizer_load_json(arr, strlen(arr)) == NULL);
}

/* --- G1: model.type strict whitelist ------------------------------------------ */

/* Absent model.type is a load failure: real HF exports always carry it, and
 * guessing BPE for an unlabeled model risks silently wrong ids. */
static void test_load_rejects_missing_model_type(void) {
    const char *json = "{\"model\":{\"vocab\":{\"a\":0},\"merges\":[]}}";
    assert(tokenizer_load_json(json, strlen(json)) == NULL);
}

static void test_load_rejects_wordlevel_model_type(void) {
    const char *json =
        "{\"model\":{\"type\":\"WordLevel\",\"vocab\":{\"a\":0},\"merges\":[]}}";
    assert(tokenizer_load_json(json, strlen(json)) == NULL);
}

static void test_load_accepts_bpe_type(void) {
    const char *json = "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":0},\"merges\":[]}}";
    tokenizer_t *tok = tokenizer_load_json(json, strlen(json));
    assert(tok != NULL);
    tokenizer_free(tok);
}

static void test_load_accepts_wordpiece_type(void) {
    const char *json = "{\"model\":{\"type\":\"WordPiece\",\"vocab\":{\"a\":0}}}";
    tokenizer_t *tok = tokenizer_load_json(json, strlen(json));
    assert(tok != NULL);
    tokenizer_free(tok);
}

/* --- G3: ALL added_tokens entries are stored (special and non-special) --------- */

/* Non-special added tokens like <think> / <tool_call> must still match
 * atomically under parse_special=true (Zig-faithful single map). */
static void test_added_tokens_non_special_stored(void) {
    const char *json =
        "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"x\":1,\"y\":2},\"merges\":[]},"
        "\"added_tokens\":[{\"id\":50,\"content\":\"<tool_call>\",\"special\":false}]}";
    tokenizer_t *tok = tokenizer_load_json(json, strlen(json));
    assert(tok != NULL);

    int32_t *ids;
    int      n = tokenizer_encode_alloc(tok, "x<tool_call>y", 13, true, &ids);
    assert(n == 3);
    assert(ids[0] == 1);
    assert(ids[1] == 50);
    assert(ids[2] == 2);
    free(ids);

    tokenizer_free(tok);
}

/* A missing "special" field is not a filter: the entry is still stored. */
static void test_added_tokens_missing_special_field_stored(void) {
    const char *json =
        "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"x\":1},\"merges\":[]},"
        "\"added_tokens\":[{\"id\":50,\"content\":\"<tool_call>\"}]}";
    tokenizer_t *tok = tokenizer_load_json(json, strlen(json));
    assert(tok != NULL);

    int32_t *ids;
    int      n = tokenizer_encode_alloc(tok, "<tool_call>", 11, true, &ids);
    assert(n == 1);
    assert(ids[0] == 50);
    free(ids);

    tokenizer_free(tok);
}

/* Added tokens absent from model.vocab join the vocab: they decode back to
 * their content and count toward vocab_size. */
static void test_added_tokens_added_to_vocab_if_missing(void) {
    const char *json =
        "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"x\":1},\"merges\":[]},"
        "\"added_tokens\":[{\"id\":50,\"content\":\"<tool_call>\",\"special\":true}]}";
    tokenizer_t *tok = tokenizer_load_json(json, strlen(json));
    assert(tok != NULL);
    assert(tokenizer_vocab_size(tok) == 2);
    assert(strcmp(tokenizer_decode_token(tok, 50), "<tool_call>") == 0);
    tokenizer_free(tok);
}

/* Added tokens already in model.vocab do not duplicate: vocab_size is
 * unchanged and the id still decodes. */
static void test_added_tokens_already_in_vocab_no_dup(void) {
    const char *json =
        "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"x\":1,\"<s>\":2},\"merges\":[]},"
        "\"added_tokens\":[{\"id\":2,\"content\":\"<s>\",\"special\":true}]}";
    tokenizer_t *tok = tokenizer_load_json(json, strlen(json));
    assert(tok != NULL);
    assert(tokenizer_vocab_size(tok) == 2);
    assert(strcmp(tokenizer_decode_token(tok, 2), "<s>") == 0);
    tokenizer_free(tok);
}

/* --- G4: BOS/EOS resolution via ordered name lists ------------------------------ */

static void test_bos_eos_sp_names(void) {
    const char *json =
        "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"<s>\":1,\"</s>\":2,\"a\":3},"
        "\"merges\":[]},"
        "\"added_tokens\":[{\"id\":1,\"content\":\"<s>\",\"special\":true},"
        "{\"id\":2,\"content\":\"</s>\",\"special\":true}]}";
    tokenizer_t *tok = tokenizer_load_json(json, strlen(json));
    assert(tok != NULL);
    assert(tokenizer_bos_id(tok) == 1);
    assert(tokenizer_eos_id(tok) == 2);
    tokenizer_free(tok);
}

static void test_bos_eos_llama3_names(void) {
    const char *json =
        "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":0},\"merges\":[]},"
        "\"added_tokens\":[{\"id\":7,\"content\":\"<|begin_of_text|>\",\"special\":true},"
        "{\"id\":8,\"content\":\"<|end_of_text|>\",\"special\":true}]}";
    tokenizer_t *tok = tokenizer_load_json(json, strlen(json));
    assert(tok != NULL);
    assert(tokenizer_bos_id(tok) == 7);
    assert(tokenizer_eos_id(tok) == 8);
    tokenizer_free(tok);
}

/* The name lists are ordered: <bos> outranks <s> when both are present. */
static void test_bos_eos_first_match_wins(void) {
    const char *json =
        "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":0},\"merges\":[]},"
        "\"added_tokens\":[{\"id\":11,\"content\":\"<s>\",\"special\":true},"
        "{\"id\":10,\"content\":\"<bos>\",\"special\":true}]}";
    tokenizer_t *tok = tokenizer_load_json(json, strlen(json));
    assert(tok != NULL);
    assert(tokenizer_bos_id(tok) == 10);
    tokenizer_free(tok);
}

/* GPT-2 has only <|endoftext|> (in its added_tokens): eos resolves, bos
 * stays -1. */
static void test_bos_eos_gpt2_fixture(void) {
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
    assert(tokenizer_bos_id(tok) == -1);
    assert(tokenizer_eos_id(tok) == 50256);
    tokenizer_free(tok);
    free(buf);
}

/* WordPiece regression: [CLS]/[SEP] added tokens keep resolving through the
 * generalized scan. */
static void test_bos_eos_wordpiece_regression(void) {
    const char *json =
        "{\"model\":{\"type\":\"WordPiece\",\"vocab\":{\"[CLS]\":101,\"[SEP]\":102,"
        "\"[UNK]\":0,\"hello\":10}},"
        "\"added_tokens\":[{\"id\":101,\"content\":\"[CLS]\",\"special\":true},"
        "{\"id\":102,\"content\":\"[SEP]\",\"special\":true}]}";
    tokenizer_t *tok = tokenizer_load_json(json, strlen(json));
    assert(tok != NULL);
    assert(tokenizer_bos_id(tok) == 101);
    assert(tokenizer_eos_id(tok) == 102);
    tokenizer_free(tok);
}

/* Names present ONLY in model.vocab (no added_tokens at all) still resolve:
 * the scan falls back from special_tokens to vocab. */
static void test_bos_eos_vocab_fallback(void) {
    const char *json =
        "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"<s>\":5,\"</s>\":6,\"a\":7},"
        "\"merges\":[]}}";
    tokenizer_t *tok = tokenizer_load_json(json, strlen(json));
    assert(tok != NULL);
    assert(tokenizer_bos_id(tok) == 5);
    assert(tokenizer_eos_id(tok) == 6);
    tokenizer_free(tok);
}

/* --- G5: malformed tokenizer.json rejection (table-driven) ---------------------- */

static void test_malformed_json_table_driven(void) {
    static const char *cases[] = {
        /* 1: root not object */
        "[]",
        /* 2: model missing */
        "{}",
        /* 3: model not object */
        "{\"model\":42}",
        /* 4: vocab missing */
        "{\"model\":{\"type\":\"BPE\"}}",
        /* 5: vocab not object */
        "{\"model\":{\"type\":\"BPE\",\"vocab\":42}}",
        /* 6: vocab value not int */
        "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":\"x\"}}}",
        /* 7: vocab id negative */
        "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":-1}}}",
        /* 8: merges not array */
        "{\"model\":{\"type\":\"BPE\",\"vocab\":{},\"merges\":42}}",
        /* 9: added_tokens not array */
        "{\"model\":{\"type\":\"BPE\",\"vocab\":{},\"merges\":[]},"
        "\"added_tokens\":42}",
        /* 10: added_tokens entry not object */
        "{\"model\":{\"type\":\"BPE\",\"vocab\":{},\"merges\":[]},"
        "\"added_tokens\":[42]}",
        /* 11: content missing */
        "{\"model\":{\"type\":\"BPE\",\"vocab\":{},\"merges\":[]},"
        "\"added_tokens\":[{\"id\":1}]}",
        /* 12: content not string */
        "{\"model\":{\"type\":\"BPE\",\"vocab\":{},\"merges\":[]},"
        "\"added_tokens\":[{\"id\":1,\"content\":42}]}",
        /* 13: id missing */
        "{\"model\":{\"type\":\"BPE\",\"vocab\":{},\"merges\":[]},"
        "\"added_tokens\":[{\"content\":\"x\"}]}",
        /* 14: id not int */
        "{\"model\":{\"type\":\"BPE\",\"vocab\":{},\"merges\":[]},"
        "\"added_tokens\":[{\"content\":\"x\",\"id\":\"y\"}]}",
        /* 15: id out of range (1 << 24, above the 4M cap) */
        "{\"model\":{\"type\":\"BPE\",\"vocab\":{},\"merges\":[]},"
        "\"added_tokens\":[{\"content\":\"x\",\"id\":16777216}]}",
        /* 16: Sequence pre_tokenizer with non-array pretokenizers */
        "{\"model\":{\"type\":\"BPE\",\"vocab\":{},\"merges\":[]},"
        "\"pre_tokenizer\":{\"type\":\"Sequence\",\"pretokenizers\":42}}",
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(*cases); i++) {
        if (tokenizer_load_json(cases[i], strlen(cases[i])) != NULL) {
            fprintf(stderr, "malformed case %zu unexpectedly loaded: %s\n", i + 1,
                    cases[i]);
            assert(0 && "malformed tokenizer.json accepted");
        }
    }
}

/* --- G6: tokenizer_load from a file path ---------------------------------------- */

static void test_tokenizer_load_gpt2(void) {
    tokenizer_t *tok = tokenizer_load(MLXD_FIXTURES_DIR "/gpt2/tokenizer.json");
    assert(tok != NULL);
    assert(tokenizer_vocab_size(tok) == 50257);
    tokenizer_free(tok);
}

static void test_tokenizer_load_bert(void) {
    tokenizer_t *tok = tokenizer_load(MLXD_FIXTURES_DIR "/bert/tokenizer.json");
    assert(tok != NULL);
    tokenizer_free(tok);
}

static void test_tokenizer_load_nonexistent(void) {
    assert(tokenizer_load("/nonexistent/path.json") == NULL);
}

static void test_tokenizer_load_devnull(void) {
    assert(tokenizer_load("/dev/null") == NULL);
}

/* --- G7: tokenizer_load_dir - tokenizer.json preferred, legacy fallback ---------- */

static void test_load_dir_gpt2(void) {
    tokenizer_t *tok = tokenizer_load_dir(MLXD_FIXTURES_DIR "/gpt2");
    assert(tok != NULL);
    assert(tokenizer_vocab_size(tok) == 50257);
    assert(tokenizer_bos_id(tok) == -1);
    assert(tokenizer_eos_id(tok) == 50256);

    /* Encode/decode round-trip is lossless through the dir loader. */
    int32_t *ids;
    int      n = tokenizer_encode_alloc(tok, "hello world", 11, true, &ids);
    assert(n > 0);
    char *text = tokenizer_decode(tok, ids, n);
    assert(text != NULL);
    assert(strcmp(text, "hello world") == 0);
    free(text);
    free(ids);

    tokenizer_free(tok);
}

static void test_load_dir_bert(void) {
    tokenizer_t *tok = tokenizer_load_dir(MLXD_FIXTURES_DIR "/bert");
    assert(tok != NULL);
    tokenizer_free(tok);
}

static void test_load_dir_nonexistent(void) {
    assert(tokenizer_load_dir("/nonexistent/dir") == NULL);
}

/* Legacy vocab.json + merges.txt (no tokenizer.json): loads, merges work,
 * and encoding " a" to the Ġa id proves byte-level classification - a
 * SentencePiece tokenizer would map the space to U+2581, never Ġ. */
static void test_load_dir_legacy_fallback(void) {
    tokenizer_t *tok = tokenizer_load_dir(MLXD_FIXTURES_DIR "/legacy_bpe");
    assert(tok != NULL);
    assert(tokenizer_vocab_size(tok) == 5);

    int32_t *ids;
    int      n = tokenizer_encode_alloc(tok, "ab", 2, true, &ids);
    assert(n == 1);
    assert(ids[0] == 2);
    free(ids);

    n = tokenizer_encode_alloc(tok, " a", 2, true, &ids);
    assert(n == 1);
    assert(ids[0] == 3);
    free(ids);

    tokenizer_free(tok);
}

/* A dir whose tokenizer.json EXISTS but is corrupt must fail loudly, even
 * with valid legacy files beside it: fallback is existence-based only. */
static void test_load_dir_corrupt_tokenizer_json_fails(void) {
    assert(tokenizer_load_dir(MLXD_FIXTURES_DIR "/corrupt_json") == NULL);
}

/* --- G8: normalizer recognition -------------------------------------------------- */

static void test_normalizer_prepend_sets_flag(void) {
    const char *json =
        "{\"normalizer\":{\"type\":\"Prepend\",\"prepend\":\"\xe2\x96\x81\"},"
        "\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":0},\"merges\":[]}}";
    tokenizer_t *tok = tokenizer_load_json(json, strlen(json));
    assert(tok != NULL);
    assert(tokenizer_add_dummy_prefix(tok));
    tokenizer_free(tok);
}

static void test_normalizer_replace_is_noop(void) {
    const char *json =
        "{\"normalizer\":{\"type\":\"Replace\",\"pattern\":{\"String\":\" \"},"
        "\"content\":\"\xe2\x96\x81\"},"
        "\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":0},\"merges\":[]}}";
    tokenizer_t *tok = tokenizer_load_json(json, strlen(json));
    assert(tok != NULL);
    assert(!tokenizer_add_dummy_prefix(tok));
    tokenizer_free(tok);
}

static void test_normalizer_unknown_loads_ok(void) {
    const char *json =
        "{\"normalizer\":{\"type\":\"UnknownXyz\"},"
        "\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":0},\"merges\":[]}}";
    tokenizer_t *tok = tokenizer_load_json(json, strlen(json));
    assert(tok != NULL);
    assert(!tokenizer_add_dummy_prefix(tok));
    tokenizer_free(tok);
}

static void test_normalizer_sequence_with_prepend(void) {
    const char *json =
        "{\"normalizer\":{\"type\":\"Sequence\",\"normalizers\":["
        "{\"type\":\"Replace\",\"pattern\":{\"String\":\" \"},"
        "\"content\":\"\xe2\x96\x81\"},"
        "{\"type\":\"Prepend\",\"prepend\":\"\xe2\x96\x81\"}]},"
        "\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":0},\"merges\":[]}}";
    tokenizer_t *tok = tokenizer_load_json(json, strlen(json));
    assert(tok != NULL);
    assert(tokenizer_add_dummy_prefix(tok));
    tokenizer_free(tok);
}

static void test_normalizer_null_or_absent(void) {
    const char *null_json =
        "{\"normalizer\":null,"
        "\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":0},\"merges\":[]}}";
    tokenizer_t *tok = tokenizer_load_json(null_json, strlen(null_json));
    assert(tok != NULL);
    assert(!tokenizer_add_dummy_prefix(tok));
    tokenizer_free(tok);

    const char *absent_json =
        "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":0},\"merges\":[]}}";
    tok = tokenizer_load_json(absent_json, strlen(absent_json));
    assert(tok != NULL);
    assert(!tokenizer_add_dummy_prefix(tok));
    tokenizer_free(tok);
}

/* --- G9: BOS/EOS overrides from tokenizer_config.json ---------------------------- */

/* String form: config names beat the G4 heuristic (<s>/</s> would win it). */
static void test_bos_eos_config_string_form(void) {
    tokenizer_t *tok = tokenizer_load_dir(MLXD_FIXTURES_DIR "/config_override");
    assert(tok != NULL);
    assert(tokenizer_bos_id(tok) == 10);
    assert(tokenizer_eos_id(tok) == 11);
    tokenizer_free(tok);
}

/* Object form: {"bos_token":{"content":"..."}}. */
static void test_bos_eos_config_object_form(void) {
    tokenizer_t *tok = tokenizer_load_dir(MLXD_FIXTURES_DIR "/config_object");
    assert(tok != NULL);
    assert(tokenizer_bos_id(tok) == 10);
    assert(tokenizer_eos_id(tok) == 11);
    tokenizer_free(tok);
}

/* No tokenizer_config.json in the dir: the G4 heuristic values stand. */
static void test_bos_eos_config_absent_uses_heuristic(void) {
    tokenizer_t *tok = tokenizer_load_dir(MLXD_FIXTURES_DIR "/bert");
    assert(tok != NULL);
    assert(tokenizer_bos_id(tok) == 101);
    assert(tokenizer_eos_id(tok) == 102);
    tokenizer_free(tok);
}

/* Config naming tokens absent from the vocab: override skipped, heuristic
 * result kept, no crash. */
static void test_bos_eos_config_unknown_token(void) {
    tokenizer_t *tok = tokenizer_load_dir(MLXD_FIXTURES_DIR "/config_unknown");
    assert(tok != NULL);
    assert(tokenizer_bos_id(tok) == 1);
    assert(tokenizer_eos_id(tok) == 2);
    tokenizer_free(tok);
}

/* --- G2: parse_merge_pair dual on-disk formats -------------------------------- */

/* Merges as space-joined strings ("a b", GPT-2-style exports) must build the
 * same merge table as the 2-string array form. */
static void test_parse_merge_string_format(void) {
    const char *json =
        "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":1,\"b\":2,\"ab\":3},"
        "\"merges\":[\"a b\"]}}";
    tokenizer_t *tok = tokenizer_load_json(json, strlen(json));
    assert(tok != NULL);

    encode_scratch s;
    encode_scratch_init(&s);
    assert(encode_scratch_reserve(&s, 2));

    int32_t *out   = NULL;
    int      count = bpe_merge(tok, &s, "ab", 2, &out);
    assert(count == 1);
    assert(out[0] == 3);

    encode_scratch_free(&s);
    tokenizer_free(tok);
}

/* Malformed merge entries (integer, null, single-element array, string with
 * no space) are silently skipped; valid entries around them still apply. */
static void test_parse_merge_skips_malformed(void) {
    const char *json =
        "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":1,\"b\":2,\"ab\":3},"
        "\"merges\":[42,null,[\"a\"],\"\",\"a b\"]}}";
    tokenizer_t *tok = tokenizer_load_json(json, strlen(json));
    assert(tok != NULL);

    encode_scratch s;
    encode_scratch_init(&s);
    assert(encode_scratch_reserve(&s, 2));

    int32_t *out   = NULL;
    int      count = bpe_merge(tok, &s, "ab", 2, &out);
    assert(count == 1);
    assert(out[0] == 3);

    encode_scratch_free(&s);
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

/* heap_cap = 3 * len is computed in uint32_t: any len above UINT32_MAX / 3
 * would wrap the stored capacity, so reserve must refuse it outright and
 * leave a fresh scratch untouched. */
static void test_scratch_reserve_rejects_len_overflowing_heap_cap(void) {
    encode_scratch s;
    encode_scratch_init(&s);

    assert(!encode_scratch_reserve(&s, (size_t)UINT32_MAX / 3 + 1));
    assert(s.nodes_cap == 0);
    assert(s.ids_cap == 0);
    assert(s.heap_cap == 0);

    encode_scratch_free(&s);
}

/* len used to be truncated to uint32_t inside bpe_merge while the loop
 * compared pos < len in size_t: pos stopped advancing and n_nodes overran the
 * scratch. The guard must run before any input byte is read. */
static void test_bpe_merge_rejects_oversized_len(void) {
    const char *json = "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":1,\"b\":2},\"merges\":[]}}";
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
        "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":1,\"b\":2,\"c\":3,\"d\":4,\"ab\":5,\"cd\":6,\"abcd\":7},"
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
    const char *json = "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":1,\"aa\":2},\"merges\":[[\"a\",\"a\"]]}}";
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
    const char *json = "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":1,\"aa\":2},\"merges\":[[\"a\",\"a\"]]}}";
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
        "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"h\":1,\"e\":2,\"l\":3,\"o\":4,\"he\":5,\"ll\":6,\"hell\":7,"
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
        "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":1,\"b\":2,\"c\":3,\"bc\":4,\"abc\":5},"
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
        "\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":1,\"z\":99},\"merges\":[]}}";
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

/* Byte-level fallback: input "\xC3\xA9" is ONE UTF-8 char ("é"), so it forms
 * a single symbol that misses the vocab. The fallback must emit one token per
 * RAW byte, keyed by the UTF-8 form of that byte's GPT-2 unicode mapping -
 * which differs from the raw byte string for any byte >= 0x80. */
static void test_bpe_bytelevel_fallback(void) {
    uc_bytes_unicode_t table;
    uc_build_bytes_to_unicode(&table);

    char key_c3[5] = {0};
    char key_a9[5] = {0};
    uc_encode_codepoint(table.byte_to_cp[0xC3], key_c3);
    uc_encode_codepoint(table.byte_to_cp[0xA9], key_a9);

    char json[256];
    int  n = snprintf(json, sizeof(json),
                      "{\"pre_tokenizer\":{\"type\":\"ByteLevel\"},"
                      "\"model\":{\"type\":\"BPE\",\"vocab\":{\"%s\":50,\"%s\":51},\"merges\":[]}}",
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
        "\"model\":{\"type\":\"BPE\",\"vocab\":{\"<unk>\":7},\"merges\":[]}}";
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
        "\"model\":{\"type\":\"BPE\",\"vocab\":{\"z\":1},\"merges\":[]}}";
    tok = tokenizer_load_json(json_no_unk, strlen(json_no_unk));
    assert(tok != NULL);

    assert(bpe_merge(tok, &s, "a", 1, &out) == 0);

    encode_scratch_free(&s);
    tokenizer_free(tok);
}

/* SentencePiece byte fallback: an unknown byte becomes its <0xNN> token. */
static void test_bpe_sp_hexbyte(void) {
    const char *json = "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":12,\"<0x0A>\":10},\"merges\":[]}}";
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
    const char *json = "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":12,\"<unk>\":0},\"merges\":[]}}";
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
    test_load_json_sparse_unordered_ids();
    test_load_json_gpt2_smoke();
    test_load_rejects_negative_id();
    test_load_rejects_huge_id();
    test_load_rejects_non_integer_id();
    test_load_rejects_unigram_model_type();
    test_load_rejects_non_string_model_type();
    test_load_rejects_missing_model_type();
    test_load_rejects_wordlevel_model_type();
    test_load_accepts_bpe_type();
    test_load_accepts_wordpiece_type();
    test_added_tokens_non_special_stored();
    test_added_tokens_missing_special_field_stored();
    test_added_tokens_added_to_vocab_if_missing();
    test_added_tokens_already_in_vocab_no_dup();
    test_bos_eos_sp_names();
    test_bos_eos_llama3_names();
    test_bos_eos_first_match_wins();
    test_bos_eos_gpt2_fixture();
    test_bos_eos_wordpiece_regression();
    test_bos_eos_vocab_fallback();
    test_malformed_json_table_driven();
    test_tokenizer_load_gpt2();
    test_tokenizer_load_bert();
    test_tokenizer_load_nonexistent();
    test_tokenizer_load_devnull();
    test_load_dir_gpt2();
    test_load_dir_bert();
    test_load_dir_nonexistent();
    test_load_dir_legacy_fallback();
    test_load_dir_corrupt_tokenizer_json_fails();
    test_normalizer_prepend_sets_flag();
    test_normalizer_replace_is_noop();
    test_normalizer_unknown_loads_ok();
    test_normalizer_sequence_with_prepend();
    test_normalizer_null_or_absent();
    test_bos_eos_config_string_form();
    test_bos_eos_config_object_form();
    test_bos_eos_config_absent_uses_heuristic();
    test_bos_eos_config_unknown_token();
    test_parse_merge_string_format();
    test_parse_merge_skips_malformed();
    test_reserve_rejects_oversized_input();
    test_scratch_reserve_rejects_len_overflowing_heap_cap();
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
