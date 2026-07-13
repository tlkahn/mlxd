/* Stage F: special-token scan + public encode entry points.
 * Expected ids mirror the Zig reference tests (tokenizer.zig sp_json). */

#include "model/tokenizer.h"

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* SentencePiece-style fixture shared with the Zig sp_json: no pre_tokenizer,
 * type BPE, so it loads as TOKENIZER_SENTENCEPIECE_BPE. */
static tokenizer_t *load_sp_tok(void) {
    const char *json =
        "{\"model\":{\"type\":\"BPE\",\"vocab\":{"
        "\"<unk>\":0,\"<s>\":1,\"</s>\":2,"
        "\"\xe2\x96\x81\":3,\"\xe2\x96\x81hello\":4,\"hello\":5,"
        "\"<0x0A>\":10,\"<0x41>\":11,\"a\":12},\"merges\":[]},"
        "\"added_tokens\":["
        "{\"id\":1,\"content\":\"<s>\",\"special\":true},"
        "{\"id\":2,\"content\":\"</s>\",\"special\":true},"
        "{\"id\":20,\"content\":\"<|im_end|>\",\"special\":true}]}";
    tokenizer_t *tok = tokenizer_load_json(json, strlen(json));
    assert(tok != NULL);
    return tok;
}

/* --- F1: parse_special=true splits on added special tokens ------------------- */

static void test_parse_special_true(void) {
    tokenizer_t *tok = load_sp_tok();

    /* A lone control marker collapses to its single special id. */
    int32_t *ids;
    int      n = tokenizer_encode_alloc(tok, "<|im_end|>", 10, true, &ids);
    assert(n == 1);
    assert(ids[0] == 20);
    free(ids);

    /* Ordinary text before the marker encodes per mode; the marker's id is
     * emitted atomically at the end. */
    n = tokenizer_encode_alloc(tok, "hello<|im_end|>", 15, true, &ids);
    assert(n >= 2);
    assert(ids[n - 1] == 20);
    for (int i = 0; i < n - 1; i++) assert(ids[i] != 20);
    free(ids);

    tokenizer_free(tok);
}

/* --- F2: parse_special=false treats control markers as plain text ------------- */

static void test_parse_special_false(void) {
    tokenizer_t *tok = load_sp_tok();

    /* The same marker encodes as ordinary text: every byte falls back to
     * <0xNN>/unk ids, never the control id 20. */
    int32_t *ids;
    int      n = tokenizer_encode_alloc(tok, "<|im_end|>", 10, false, &ids);
    assert(n > 1);
    for (int i = 0; i < n; i++) assert(ids[i] != 20);
    free(ids);

    tokenizer_free(tok);
}

/* --- F3: tie-break - earliest match wins, ties go to the longest key ----------- */

static void test_special_tie_break(void) {
    const char *json =
        "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"<unk>\":0,\"x\":30,\"y\":31},"
        "\"merges\":[]},"
        "\"added_tokens\":["
        "{\"id\":40,\"content\":\"<|im\",\"special\":true},"
        "{\"id\":41,\"content\":\"<|im_end|>\",\"special\":true},"
        "{\"id\":42,\"content\":\"<|end|>\",\"special\":true}]}";
    tokenizer_t *tok = tokenizer_load_json(json, strlen(json));
    assert(tok != NULL);

    /* "<|im" and "<|im_end|>" both match at index 1: the longer key wins.
     * "<|im" occurs again at index 12 (after the consumed match) and wins
     * there; "<|end|>" closes. Mirrors tokenizer.zig's tie-break guard. */
    int32_t *ids;
    int      n = tokenizer_encode_alloc(tok, "x<|im_end|>y<|im<|end|>", 23, true, &ids);
    assert(n >= 3);
    int32_t specials[8];
    int     n_sp = 0;
    for (int i = 0; i < n; i++) {
        if (ids[i] >= 40) {
            assert(n_sp < 8);
            specials[n_sp++] = ids[i];
        }
    }
    assert(n_sp == 3);
    assert(specials[0] == 41);
    assert(specials[1] == 40);
    assert(specials[2] == 42);
    free(ids);

    tokenizer_free(tok);
}

/* --- F4: WordPiece [CLS]/[SEP] wrap lives in the entry point ------------------- */

static void test_wordpiece_wrap_in_entry(void) {
    const char *json =
        "{\"model\":{\"type\":\"WordPiece\",\"vocab\":{\"[CLS]\":101,\"[SEP]\":102,"
        "\"[UNK]\":0,\"hello\":10,\"world\":11,\"hel\":12,\"##lo\":13}},"
        "\"added_tokens\":[{\"id\":101,\"content\":\"[CLS]\",\"special\":true},"
        "{\"id\":102,\"content\":\"[SEP]\",\"special\":true}]}";
    tokenizer_t *tok = tokenizer_load_json(json, strlen(json));
    assert(tok != NULL);

    /* The wrap is added by the tokenizer, not parsed from input, so
     * parse_special=false must NOT drop it (Zig EncodeOptions doc). */
    int32_t *ids;
    int      n = tokenizer_encode_alloc(tok, "hello world", 11, false, &ids);
    assert(n == 4);
    assert(ids[0] == 101);
    assert(ids[1] == 10);
    assert(ids[2] == 11);
    assert(ids[3] == 102);
    free(ids);

    n = tokenizer_encode_alloc(tok, "hello world", 11, true, &ids);
    assert(n == 4);
    assert(ids[0] == 101);
    assert(ids[3] == 102);
    free(ids);

    tokenizer_free(tok);
}

/* --- F5: tokenizer_encode truncation contract (snprintf-style) ----------------- */

static void test_encode_truncation(void) {
    tokenizer_t *tok = load_sp_tok();

    /* Learn the full encoding via the alloc entry (parse_special=true, same
     * as the fixed-buffer entry always uses). */
    int32_t *full;
    int      n_full = tokenizer_encode_alloc(tok, "hello<|im_end|>", 15, true, &full);
    assert(n_full >= 3);

    /* max_ids one short: returns the FULL count, writes exactly max_ids ids,
     * and never touches past the end (sentinel survives). */
    int32_t buf[32];
    assert(n_full + 1 <= (int)(sizeof(buf) / sizeof(buf[0])));
    for (int i = 0; i < n_full + 1; i++) buf[i] = -7;
    int n = tokenizer_encode(tok, "hello<|im_end|>", buf, n_full - 1);
    assert(n == n_full);
    for (int i = 0; i < n_full - 1; i++) assert(buf[i] == full[i]);
    assert(buf[n_full - 1] == -7);
    assert(buf[n_full] == -7);

    /* max_ids == full count round-trips exactly. */
    for (int i = 0; i < n_full + 1; i++) buf[i] = -7;
    n = tokenizer_encode(tok, "hello<|im_end|>", buf, n_full);
    assert(n == n_full);
    for (int i = 0; i < n_full; i++) assert(buf[i] == full[i]);
    assert(buf[n_full] == -7);

    /* NULL args are an error, not a crash. */
    assert(tokenizer_encode(NULL, "x", buf, 8) == -1);
    assert(tokenizer_encode(tok, NULL, buf, 8) == -1);
    assert(tokenizer_encode(tok, "x", NULL, 8) == -1);

    free(full);
    tokenizer_free(tok);
}

/* --- F6: empty input encodes to 0 ids, not an error ---------------------------- */

/* Characterization guard (already green after F1-F5, not a bug fix): pins
 * count 0 with *out_ids NULL for the alloc entry and 0 for the fixed-buffer
 * entry, under both parse_special values. */
static void test_empty_input(void) {
    tokenizer_t *tok = load_sp_tok();

    int32_t *ids = (int32_t *)&ids; /* poison: must come back NULL */
    assert(tokenizer_encode_alloc(tok, "", 0, true, &ids) == 0);
    assert(ids == NULL);
    ids = (int32_t *)&ids;
    assert(tokenizer_encode_alloc(tok, "", 0, false, &ids) == 0);
    assert(ids == NULL);

    int32_t buf[8] = {-7};
    assert(tokenizer_encode(tok, "", buf, 8) == 0);
    assert(buf[0] == -7);

    tokenizer_free(tok);
}

/* --- F7: WordPiece without [CLS]/[SEP] on empty input -------------------------- */

/* Regression: with bos_id/eos_id unresolved and an empty body, every
 * idbuf_append is n==0 on a NULL buffer; the old code reached
 * memcpy(NULL + 0, ...), UB that UBSan flags as "applying zero offset to
 * null pointer". Must return count 0 with *out_ids NULL. */
static void test_wordpiece_no_cls_sep_empty(void) {
    const char *json =
        "{\"model\":{\"type\":\"WordPiece\",\"vocab\":{\"[UNK]\":0,"
        "\"hello\":10,\"world\":11}}}";
    tokenizer_t *tok = tokenizer_load_json(json, strlen(json));
    assert(tok != NULL);

    int32_t *ids = (int32_t *)&ids; /* poison: must come back NULL */
    assert(tokenizer_encode_alloc(tok, "", 0, true, &ids) == 0);
    assert(ids == NULL);
    ids = (int32_t *)&ids;
    assert(tokenizer_encode_alloc(tok, "", 0, false, &ids) == 0);
    assert(ids == NULL);

    /* Non-empty input still encodes, just without the wrap. */
    int n = tokenizer_encode_alloc(tok, "hello world", 11, true, &ids);
    assert(n == 2);
    assert(ids[0] == 10);
    assert(ids[1] == 11);
    free(ids);

    tokenizer_free(tok);
}

/* --- F8: empty-content special is skipped at load ------------------------------- */

/* Regression: an added_tokens entry with content "" can never match input;
 * the loader must skip it (not fail, not store a dead probe) and encoding
 * must never emit its id. */
static void test_empty_content_special_skipped(void) {
    const char *json =
        "{\"model\":{\"type\":\"BPE\",\"vocab\":{"
        "\"<unk>\":0,\"\xe2\x96\x81\":3,\"\xe2\x96\x81hello\":4,\"hello\":5},"
        "\"merges\":[]},"
        "\"added_tokens\":["
        "{\"id\":99,\"content\":\"\",\"special\":true},"
        "{\"id\":20,\"content\":\"<|im_end|>\",\"special\":true}]}";
    tokenizer_t *tok = tokenizer_load_json(json, strlen(json));
    assert(tok != NULL);

    int32_t *ids;
    int      n = tokenizer_encode_alloc(tok, "hello", 5, true, &ids);
    assert(n >= 1);
    for (int i = 0; i < n; i++) assert(ids[i] != 99);
    free(ids);

    /* A real special alongside the empty one still resolves. */
    n = tokenizer_encode_alloc(tok, "hello<|im_end|>", 15, true, &ids);
    assert(n >= 2);
    assert(ids[n - 1] == 20);
    for (int i = 0; i < n; i++) assert(ids[i] != 99);
    free(ids);

    tokenizer_free(tok);
}

int main(void) {
    test_parse_special_true();
    test_parse_special_false();
    test_special_tie_break();
    test_wordpiece_wrap_in_entry();
    test_encode_truncation();
    test_empty_input();
    test_wordpiece_no_cls_sep_empty();
    test_empty_content_special_skipped();
    printf("test_tok_special: all tests passed\n");
    return 0;
}
