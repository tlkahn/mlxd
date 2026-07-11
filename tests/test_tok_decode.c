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

int main(void) {
    test_byte_level_decode();
    printf("test_tok_decode: all tests passed\n");
    return 0;
}
