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

int main(void) {
    test_byte_level_encode();
    printf("test_tok_encode: all tests passed\n");
    return 0;
}
