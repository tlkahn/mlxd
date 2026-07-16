#include "model/detok.h"
#include "model/tokenizer.h"

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef MLXD_FIXTURES_DIR
#error "MLXD_FIXTURES_DIR must be defined"
#endif

static bool is_valid_utf8(const char *buf, size_t len) {
    size_t i = 0;
    while (i < len) {
        unsigned char c = (unsigned char)buf[i];
        size_t seq;
        if (c < 0x80) { seq = 1; }
        else if ((c & 0xE0) == 0xC0) { seq = 2; }
        else if ((c & 0xF0) == 0xE0) { seq = 3; }
        else if ((c & 0xF8) == 0xF0) { seq = 4; }
        else { return false; }
        if (i + seq > len) return false;
        for (size_t j = 1; j < seq; j++) {
            if (((unsigned char)buf[i + j] & 0xC0) != 0x80)
                return false;
        }
        i += seq;
    }
    return true;
}

static void test_detok_ascii_roundtrip(void) {
    tokenizer_t *tok = tokenizer_load(MLXD_FIXTURES_DIR "/gpt2/tokenizer.json");
    assert(tok != NULL);

    const char *text = "hello world";
    int32_t *ids = NULL;
    int n = tokenizer_encode_alloc(tok, text, strlen(text), false, &ids);
    assert(n > 0);

    detok_t *d = detok_create(tok);
    assert(d != NULL);

    char *accum = calloc(1, 1);
    size_t accum_len = 0;
    for (int i = 0; i < n; i++) {
        char *out = NULL;
        size_t out_len = 0;
        assert(detok_feed(d, ids[i], &out, &out_len) == 0);
        assert(out != NULL);
        accum = realloc(accum, accum_len + out_len + 1);
        memcpy(accum + accum_len, out, out_len);
        accum_len += out_len;
        accum[accum_len] = '\0';
        free(out);
    }
    char *flush_out = NULL;
    size_t flush_len = 0;
    assert(detok_flush(d, &flush_out, &flush_len) == 0);
    assert(flush_out != NULL);
    accum = realloc(accum, accum_len + flush_len + 1);
    memcpy(accum + accum_len, flush_out, flush_len);
    accum_len += flush_len;
    accum[accum_len] = '\0';
    free(flush_out);

    char *full = tokenizer_decode(tok, ids, n);
    assert(full != NULL);
    assert(accum_len == strlen(full));
    assert(memcmp(accum, full, accum_len) == 0);

    free(accum);
    free(full);
    free(ids);
    detok_free(d);
    tokenizer_free(tok);
}

static void test_detok_matches_full_decode(void) {
    tokenizer_t *tok = tokenizer_load(MLXD_FIXTURES_DIR "/gpt2/tokenizer.json");
    assert(tok != NULL);

    const char *strings[] = {
        "  spaces  and   tabs\t",
        "punctuation: foo, bar; baz!",
        "mixed 123 #$% end",
    };

    for (int s = 0; s < 3; s++) {
        const char *text = strings[s];
        int32_t *ids = NULL;
        int n = tokenizer_encode_alloc(tok, text, strlen(text), false, &ids);
        assert(n > 0);

        detok_t *d = detok_create(tok);
        char *accum = calloc(1, 1);
        size_t accum_len = 0;
        for (int i = 0; i < n; i++) {
            char *out = NULL;
            size_t out_len = 0;
            assert(detok_feed(d, ids[i], &out, &out_len) == 0);
            accum = realloc(accum, accum_len + out_len + 1);
            memcpy(accum + accum_len, out, out_len);
            accum_len += out_len;
            free(out);
        }
        char *flush_out = NULL;
        size_t flush_len = 0;
        assert(detok_flush(d, &flush_out, &flush_len) == 0);
        accum = realloc(accum, accum_len + flush_len + 1);
        memcpy(accum + accum_len, flush_out, flush_len);
        accum_len += flush_len;
        accum[accum_len] = '\0';
        free(flush_out);

        char *full = tokenizer_decode(tok, ids, n);
        assert(full != NULL);
        assert(accum_len == strlen(full));
        assert(memcmp(accum, full, accum_len) == 0);

        free(accum);
        free(full);
        free(ids);
        detok_free(d);
    }
    tokenizer_free(tok);
}

static void test_detok_holds_incomplete_utf8(void) {
    tokenizer_t *tok = tokenizer_load(MLXD_FIXTURES_DIR "/gpt2/tokenizer.json");
    assert(tok != NULL);

    const char *text = "\xF0\x9F\x8E\x89"; /* U+1F389 party popper */
    int32_t *ids = NULL;
    int n = tokenizer_encode_alloc(tok, text, strlen(text), false, &ids);
    assert(n > 1);

    detok_t *d = detok_create(tok);
    char *accum = calloc(1, 1);
    size_t accum_len = 0;
    for (int i = 0; i < n; i++) {
        char *out = NULL;
        size_t out_len = 0;
        assert(detok_feed(d, ids[i], &out, &out_len) == 0);
        assert(out != NULL);
        assert(is_valid_utf8(out, out_len));
        accum = realloc(accum, accum_len + out_len + 1);
        memcpy(accum + accum_len, out, out_len);
        accum_len += out_len;
        free(out);
    }

    char *flush_out = NULL;
    size_t flush_len = 0;
    assert(detok_flush(d, &flush_out, &flush_len) == 0);
    accum = realloc(accum, accum_len + flush_len + 1);
    memcpy(accum + accum_len, flush_out, flush_len);
    accum_len += flush_len;
    accum[accum_len] = '\0';
    free(flush_out);

    char *full = tokenizer_decode(tok, ids, n);
    assert(full != NULL);
    assert(accum_len == strlen(full));
    assert(memcmp(accum, full, accum_len) == 0);

    free(accum);
    free(full);
    free(ids);
    detok_free(d);
    tokenizer_free(tok);
}

static void test_detok_flush_emits_trailing(void) {
    tokenizer_t *tok = tokenizer_load(MLXD_FIXTURES_DIR "/gpt2/tokenizer.json");
    assert(tok != NULL);

    const char *text = "\xF0\x9F\x8E\x89";
    int32_t *ids = NULL;
    int n = tokenizer_encode_alloc(tok, text, strlen(text), false, &ids);
    assert(n > 1);

    detok_t *d = detok_create(tok);
    char *out = NULL;
    size_t out_len = 0;
    assert(detok_feed(d, ids[0], &out, &out_len) == 0);
    assert(out != NULL);
    assert(out_len == 0);
    free(out);

    char *flush_out = NULL;
    size_t flush_len = 0;
    assert(detok_flush(d, &flush_out, &flush_len) == 0);
    assert(flush_out != NULL);
    assert(flush_len > 0);

    free(flush_out);
    free(ids);
    detok_free(d);
    tokenizer_free(tok);
}

static void test_detok_empty_and_free(void) {
    tokenizer_t *tok = tokenizer_load(MLXD_FIXTURES_DIR "/gpt2/tokenizer.json");
    assert(tok != NULL);
    detok_t *d = detok_create(tok);
    assert(d != NULL);

    char *out = NULL;
    size_t out_len = 0;
    assert(detok_flush(d, &out, &out_len) == 0);
    assert(out != NULL);
    assert(out_len == 0);
    assert(out[0] == '\0');
    free(out);

    detok_free(d);
    tokenizer_free(tok);
}

static void test_wordpiece_rejected(void) {
    tokenizer_t *tok = tokenizer_load(MLXD_FIXTURES_DIR "/bert/tokenizer.json");
    assert(tok != NULL);
    assert(tokenizer_type(tok) == TOKENIZER_WORDPIECE);

    detok_t *d = detok_create(tok);
    assert(d == NULL);

    tokenizer_free(tok);
}

int main(void) {
    test_detok_ascii_roundtrip();
    test_detok_matches_full_decode();
    test_detok_holds_incomplete_utf8();
    test_detok_flush_emits_trailing();
    test_detok_empty_and_free();
    test_wordpiece_rejected();
    printf("test_detok: all passed\n");
    return 0;
}
