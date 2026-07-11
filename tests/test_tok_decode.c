#include "model/tok_bpe.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* decode via the public entry point and assert exact string equality. */
static void expect_decode(const tokenizer_t *tok, const int32_t *ids, int count,
                          const char *want) {
    char *got = tokenizer_decode(tok, ids, count);
    assert(got != NULL);
    assert(strcmp(got, want) == 0);
    free(got);
}

/* --- E2: byte-level decode + round-trip -------------------------------------- */

static void test_byte_level_decode(void) {
    /* E1 vocab plus "萬" (U+842C, above the byte table's 324-cp range) to
     * exercise the unmapped-codepoint raw UTF-8 fallback. */
    const char *json =
        "{\"pre_tokenizer\":{\"type\":\"ByteLevel\"},"
        "\"model\":{\"vocab\":{\"a\":1,\"b\":2,\"\xc4\xa0\":3,\"\xc4\xa0" "b\":4,"
        "\"\xe8\x90\xac\":5},"
        "\"merges\":[[\"\xc4\xa0\",\"b\"]]}}";
    tokenizer_t *tok = tokenizer_load_json(json, strlen(json));
    assert(tok != NULL);

    /* Round-trip of E1's encode result. */
    expect_decode(tok, (const int32_t[]){1, 4}, 2, "a b");

    /* Out-of-range id is skipped, not an error. */
    expect_decode(tok, (const int32_t[]){1, 99, 4}, 3, "a b");

    /* Unmapped codepoint passes through as its raw UTF-8 bytes. */
    expect_decode(tok, (const int32_t[]){5}, 1, "\xe8\x90\xac");

    /* count == 0 decodes to the empty string, not NULL. */
    expect_decode(tok, NULL, 0, "");

    tokenizer_free(tok);
}

/* --- E9: WordPiece decode ------------------------------------------------------ */

static void test_wordpiece_decode(void) {
    const char *json =
        "{\"model\":{\"type\":\"WordPiece\",\"vocab\":{\"[CLS]\":101,\"[SEP]\":102,"
        "\"hello\":10,\"##ly\":11,\"world\":12}},"
        "\"added_tokens\":[{\"content\":\"[CLS]\",\"special\":true},"
        "{\"content\":\"[SEP]\",\"special\":true}]}";
    tokenizer_t *tok = tokenizer_load_json(json, strlen(json));
    assert(tok != NULL);

    /* Specials dropped; ## glues with no space; plain tokens get one space. */
    expect_decode(tok, (const int32_t[]){101, 10, 11, 12, 102}, 5, "helloly world");
    tokenizer_free(tok);

    /* An ordinary vocab token that merely starts with '[' is kept: only
     * registered specials are dropped. */
    const char *json2 =
        "{\"model\":{\"type\":\"WordPiece\",\"vocab\":{\"[\":1,\"hello\":2}}}";
    tok = tokenizer_load_json(json2, strlen(json2));
    assert(tok != NULL);
    expect_decode(tok, (const int32_t[]){2, 1}, 2, "hello [");
    tokenizer_free(tok);
}

/* --- E11: SentencePiece decode -------------------------------------------------- */

/* SP fixture shared by E11/E12: byte-fallback tokens, U+2581-prefixed and
 * bare word tokens, plus "<0x1>" - shaped like a byte token but the wrong
 * length, so it must render verbatim. */
static tokenizer_t *load_sp_tok(void) {
    const char *json =
        "{\"model\":{\"vocab\":{\"<unk>\":0,\"<s>\":1,\"</s>\":2,"
        "\"\xe2\x96\x81\":3,\"\xe2\x96\x81hello\":4,\"hello\":5,"
        "\"<0x0A>\":10,\"<0x41>\":11,\"a\":12,\"<0x1>\":13},\"merges\":[]}}";
    tokenizer_t *tok = tokenizer_load_json(json, strlen(json));
    assert(tok != NULL);
    return tok;
}

static void test_sentencepiece_decode(void) {
    tokenizer_t *tok = load_sp_tok();

    /* <0xNN> byte-fallback tokens emit the raw byte. */
    expect_decode(tok, (const int32_t[]){10}, 1, "\n");
    expect_decode(tok, (const int32_t[]){11}, 1, "A");

    /* U+2581 becomes ' '; the public entry never strips it. */
    expect_decode(tok, (const int32_t[]){4}, 1, " hello");

    /* strip_leading_space drops exactly one leading space. */
    char *got = decode_sentencepiece(tok, (const int32_t[]){4}, 1, true);
    assert(got != NULL);
    assert(strcmp(got, "hello") == 0);
    free(got);

    /* Wrong-length <0xN> lookalike renders verbatim. */
    expect_decode(tok, (const int32_t[]){13}, 1, "<0x1>");

    tokenizer_free(tok);
}

/* --- E12: U+2581 occupying the final 3 bytes still becomes a space -------------- */

static void test_sentencepiece_trailing_marker(void) {
    tokenizer_t *tok = load_sp_tok();
    expect_decode(tok, (const int32_t[]){5, 3}, 2, "hello ");
    expect_decode(tok, (const int32_t[]){3}, 1, " ");
    tokenizer_free(tok);
}

/* --- Capstone: real GPT-2 vocab round-trip ---------------------------------------- */

static void test_gpt2_fixture_decode(void) {
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

    /* Decode of the encode capstone's ids reproduces the input exactly. */
    expect_decode(tok, (const int32_t[]){15496, 995}, 2, "Hello world");

    tokenizer_free(tok);
    free(buf);
}

int main(void) {
    test_byte_level_decode();
    test_wordpiece_decode();
    test_sentencepiece_decode();
    test_sentencepiece_trailing_marker();
    test_gpt2_fixture_decode();
    printf("test_tok_decode: all tests passed\n");
    return 0;
}
