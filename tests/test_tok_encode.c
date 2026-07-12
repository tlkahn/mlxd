#include "model/tok_bpe.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* --- E1: byte-level encode --------------------------------------------------- */

/* "a b" pretokenizes to ["a", " b"]; the space maps to U+0120 (Ġ, \xc4\xa0)
 * via the GPT-2 byte table, and the ["Ġ","b"] merge folds " b" into one id. */
static void test_byte_level_encode(void) {
    const char *json =
        "{\"pre_tokenizer\":{\"type\":\"ByteLevel\"},"
        "\"model\":{\"vocab\":{\"a\":1,\"b\":2,\"\xc4\xa0\":3,\"\xc4\xa0" "b\":4},"
        "\"merges\":[[\"\xc4\xa0\",\"b\"]]}}";
    tokenizer_t *tok = tokenizer_load_json(json, strlen(json));
    assert(tok != NULL);

    encode_scratch s;
    encode_scratch_init(&s);

    int32_t *ids;
    int      n = encode_byte_level(tok, &s, "a b", 3, &ids);
    assert(n == 2);
    assert(ids[0] == 1);
    assert(ids[1] == 4);

    /* Empty input encodes to zero ids. */
    n = encode_byte_level(tok, &s, "", 0, &ids);
    assert(n == 0);

    encode_scratch_free(&s);
    tokenizer_free(tok);
}

/* --- E3: WordPiece encode wrap ------------------------------------------------ */

/* Loader must resolve [CLS]/[SEP] to bos/eos for WordPiece, and the encoder
 * wraps the body in them (Stage F relocates the wrap to the public entry). */
static void test_wordpiece_encode_wrap(void) {
    const char *json =
        "{\"model\":{\"type\":\"WordPiece\",\"vocab\":{\"[CLS]\":101,\"[SEP]\":102,"
        "\"[UNK]\":0,\"hello\":10,\"world\":11,\"hel\":12,\"##lo\":13}},"
        "\"added_tokens\":[{\"content\":\"[CLS]\",\"special\":true},"
        "{\"content\":\"[SEP]\",\"special\":true}]}";
    tokenizer_t *tok = tokenizer_load_json(json, strlen(json));
    assert(tok != NULL);
    assert(tokenizer_bos_id(tok) == 101);
    assert(tokenizer_eos_id(tok) == 102);

    encode_scratch s;
    encode_scratch_init(&s);

    int32_t *ids;
    int      n = encode_wordpiece(tok, &s, "hello world", 11, &ids);
    assert(n == 4);
    assert(ids[0] == 101);
    assert(ids[1] == 10);
    assert(ids[2] == 11);
    assert(ids[3] == 102);

    /* Empty input still gets the wrap. */
    n = encode_wordpiece(tok, &s, "", 0, &ids);
    assert(n == 2);
    assert(ids[0] == 101);
    assert(ids[1] == 102);

    encode_scratch_free(&s);
    tokenizer_free(tok);
}

/* Load a WordPiece tokenizer whose vocab JSON body is `vocab_body`, encode
 * `input`, and assert the ids match exactly. No [CLS]/[SEP] in these vocabs,
 * so bos/eos stay -1 and nothing is wrapped. */
static void expect_wp_encode(const char *vocab_body, const char *input, const int32_t *want,
                             int n_want) {
    char json[512];
    int  jn = snprintf(json, sizeof(json),
                       "{\"model\":{\"type\":\"WordPiece\",\"vocab\":%s}}", vocab_body);
    assert(jn > 0 && (size_t)jn < sizeof(json));
    tokenizer_t *tok = tokenizer_load_json(json, (size_t)jn);
    assert(tok != NULL);

    encode_scratch s;
    encode_scratch_init(&s);
    int32_t *ids;
    int      n = encode_wordpiece(tok, &s, input, strlen(input), &ids);
    assert(n == n_want);
    for (int i = 0; i < n_want; i++) assert(ids[i] == want[i]);
    encode_scratch_free(&s);
    tokenizer_free(tok);
}

/* --- E4: ## continuation pieces ----------------------------------------------- */

static void test_wordpiece_continuation(void) {
    expect_wp_encode("{\"[UNK]\":0,\"un\":10,\"##like\":11,\"##ly\":12}", "unlikely",
                     (const int32_t[]){10, 11, 12}, 3);
}

/* --- E5: unmatchable word emits [UNK] ------------------------------------------ */

static void test_wordpiece_unknown_word(void) {
    expect_wp_encode("{\"[UNK]\":0,\"hello\":10}", "hello xyz", (const int32_t[]){10, 0}, 2);
    /* No [UNK] in the vocab: unk falls back to id 0. */
    expect_wp_encode("{\"hello\":10}", "hello xyz", (const int32_t[]){10, 0}, 2);
}

/* An unmatchable word collapses to ONE unk even when a prefix matched: BERT
 * marks the whole word bad and replaces it, discarding partial pieces. Here
 * "hel" matches "hello" at pos 0 but "lo" has no continuation piece. */
static void test_wordpiece_whole_word_unk_collapse(void) {
    expect_wp_encode("{\"[UNK]\":0,\"hel\":1}", "hello", (const int32_t[]){0}, 1);
    /* Only the bad word collapses; the following good word still encodes. */
    expect_wp_encode("{\"[UNK]\":0,\"hel\":1,\"ok\":2}", "hello ok",
                     (const int32_t[]){0, 2}, 2);
}

/* --- E6: each punct byte is its own word ---------------------------------------- */

static void test_wordpiece_punct_split(void) {
    expect_wp_encode("{\"[UNK]\":0,\"hello\":10,\",\":11,\"world\":12}", "hello, world",
                     (const int32_t[]){10, 11, 12}, 3);
}

/* --- E7: ASCII lowercasing ------------------------------------------------------ */

static void test_wordpiece_lowercase(void) {
    expect_wp_encode("{\"[UNK]\":0,\"hello\":10}", "HELLO", (const int32_t[]){10}, 1);
}

/* --- E8: greedy longest match shrinks repeatedly -------------------------------- */

static void test_wordpiece_multi_shrink(void) {
    expect_wp_encode("{\"[UNK]\":0,\"in\":1,\"##ter\":2,\"##nation\":3,\"##al\":4}",
                     "international", (const int32_t[]){1, 2, 3, 4}, 4);
}

/* --- E10: SentencePiece encode --------------------------------------------------- */

/* No model.type and no ByteLevel pre_tokenizer: detected as SentencePiece.
 * Spaces normalize to U+2581 (\xe2\x96\x81) and the whole string is merged
 * with NO pre-tokenization, so " hi" folds into one token but "hi" cannot
 * reach any "\xe2\x96\x81"-prefixed merge. */
static void test_sentencepiece_encode(void) {
    const char *json =
        "{\"model\":{\"vocab\":{\"h\":1,\"i\":2,\"\xe2\x96\x81\":3,"
        "\"\xe2\x96\x81h\":10,\"\xe2\x96\x81hi\":4},"
        "\"merges\":[[\"\xe2\x96\x81\",\"h\"],[\"\xe2\x96\x81h\",\"i\"]]}}";
    tokenizer_t *tok = tokenizer_load_json(json, strlen(json));
    assert(tok != NULL);

    encode_scratch s;
    encode_scratch_init(&s);

    int32_t *ids;
    int      n = encode_sentencepiece(tok, &s, " hi", 3, &ids);
    assert(n == 1);
    assert(ids[0] == 4);

    n = encode_sentencepiece(tok, &s, "hi", 2, &ids);
    assert(n == 2);
    assert(ids[0] == 1);
    assert(ids[1] == 2);

    /* Empty input encodes to zero ids. */
    n = encode_sentencepiece(tok, &s, "", 0, &ids);
    assert(n == 0);

    encode_scratch_free(&s);
    tokenizer_free(tok);
}

/* --- Capstone: real GPT-2 vocab -------------------------------------------------- */

static void test_gpt2_fixture_encode(void) {
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

    encode_scratch s;
    encode_scratch_init(&s);
    int32_t *ids;
    int      n = encode_byte_level(tok, &s, "Hello world", 11, &ids);
    /* Reference ids from the HF GPT-2 tokenizer: "Hello" = 15496,
     * " world" = 995. */
    assert(n == 2);
    assert(ids[0] == 15496);
    assert(ids[1] == 995);

    encode_scratch_free(&s);
    tokenizer_free(tok);
    free(buf);
}

int main(void) {
    test_byte_level_encode();
    test_wordpiece_encode_wrap();
    test_wordpiece_continuation();
    test_wordpiece_unknown_word();
    test_wordpiece_whole_word_unk_collapse();
    test_wordpiece_punct_split();
    test_wordpiece_lowercase();
    test_wordpiece_multi_shrink();
    test_sentencepiece_encode();
    test_gpt2_fixture_encode();
    printf("test_tok_encode: all tests passed\n");
    return 0;
}
