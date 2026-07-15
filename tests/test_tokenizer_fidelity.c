/* Stage H: real-model port-fidelity gates (issue #24).
 *
 * End-to-end encode/decode verification against three real tokenizers via
 * the public API only:
 *   H1/H2  gpt2 (byte-level BPE)      - tests/fixtures/gpt2/tokenizer.json
 *   H3     bert-base-uncased (WordPiece) - tests/fixtures/bert/tokenizer.json
 *   H4     Gemma 4 (SentencePiece BPE) - tests/fixtures/gemma4/tokenizer.json
 *
 * Vector provenance:
 *   - gpt2 vectors are verbatim from the Zig reference
 *     (mlxd-zig src/model/tokenizer.zig), which took them from HF
 *     `tokenizers` on the canonical openai-community/gpt2 tokenizer.json.
 *   - bert and gemma vectors were generated with HF `tokenizers` as oracle:
 *       uv run --with tokenizers python tests/tools/gen_fidelity_vectors.py
 *     BERT with add_special_tokens=True ([CLS]/[SEP] wrap), Gemma with
 *     add_special_tokens=False (no <bos>).
 *   Oracle vectors are ground truth: a mismatch is a C bug; vectors are
 *   never adjusted to fit.
 *
 * gemma4 fixture is dev-only and NOT committed (32 MB). Setup:
 *   cp <gemma-4-E4B tokenizer.json> tests/fixtures/gemma4/tokenizer.json
 *   sha256: 12bac982b793c44b03d52a250a9f0d0b666813da566b910c24a6da0695fd11e6
 *   (vocab_size 262144). When absent, the gemma tests print [SKIP] and the
 *   suite still exits 0 so CI stays green.
 */

#include "model/tokenizer.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/* Encode text and require the exact oracle id sequence. */
static void check_vector(const tokenizer_t *tok, const char *text, const int32_t *expected,
                         int n) {
    int32_t *ids   = NULL;
    int      count = tokenizer_encode_alloc(tok, text, strlen(text), true, &ids);
    if (count != n || (n > 0 && memcmp(ids, expected, (size_t)n * sizeof(int32_t)) != 0)) {
        fprintf(stderr, "vector mismatch for %s\n  expected (%d):", text, n);
        for (int i = 0; i < n; i++) fprintf(stderr, " %d", expected[i]);
        fprintf(stderr, "\n  got (%d):", count);
        for (int i = 0; i < count; i++) fprintf(stderr, " %d", ids[i]);
        fprintf(stderr, "\n");
        assert(0 && "known-good vector mismatch");
    }
    free(ids);
}

/* Encode then decode and require the exact original bytes back. */
static void check_roundtrip(const tokenizer_t *tok, const char *text) {
    int32_t *ids   = NULL;
    int      count = tokenizer_encode_alloc(tok, text, strlen(text), true, &ids);
    assert(count >= 0);
    char *decoded = tokenizer_decode(tok, ids, count);
    assert(decoded != NULL);
    if (strcmp(decoded, text) != 0) {
        fprintf(stderr, "roundtrip mismatch\n  in:  %s\n  out: %s\n", text, decoded);
        assert(0 && "roundtrip not lossless");
    }
    free(decoded);
    free(ids);
}

/* --- Cycle 1: tokenizer_type accessor ------------------------------------ */

static void test_type_gpt2(void) {
    tokenizer_t *tok = tokenizer_load(MLXD_FIXTURES_DIR "/gpt2/tokenizer.json");
    assert(tok != NULL);
    assert(tokenizer_type(tok) == TOKENIZER_BPE);
    tokenizer_free(tok);
}

/* --- Cycle 2 (H1): gpt2 known-good vectors (verbatim from Zig reference) - */

static void test_gpt2_known_vectors(void) {
    tokenizer_t *tok = tokenizer_load(MLXD_FIXTURES_DIR "/gpt2/tokenizer.json");
    assert(tok != NULL);

    static const int32_t hello[] = {15496, 995};
    check_vector(tok, "Hello world", hello, 2);

    static const int32_t fox[] = {464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13};
    check_vector(tok, "The quick brown fox jumps over the lazy dog.", fox, 10);

    static const int32_t cjk[] = {19526, 254,   25001, 121, 171,   120, 234,
                                  10310, 244,   45911, 234, 171,   120, 223};
    check_vector(tok, "\xe4\xbd\xa0\xe5\xa5\xbd\xef\xbc\x8c\xe4\xb8\x96\xe7\x95\x8c\xef\xbc\x81",
                 cjk, 14);

    static const int32_t emoji[] = {15496, 50169, 233, 995, 12520, 234, 235, 26486, 101};
    check_vector(tok, "Hello \xf0\x9f\x91\x8b world \xf0\x9f\x8c\x8d\xe2\x9c\xa8", emoji, 9);

    static const int32_t ws[] = {220,  3756, 220, 220,  9029, 197, 392, 197, 8658,
                                 82,   198,  3605, 220, 3951, 220, 220, 220};
    check_vector(tok, "  leading   spaces\tand\ttabs\nnew  lines   ", ws, 17);

    tokenizer_free(tok);
}

/* --- Cycle 3 (H2): gpt2 lossless round-trip ------------------------------- */

static void test_gpt2_roundtrip(void) {
    tokenizer_t *tok = tokenizer_load(MLXD_FIXTURES_DIR "/gpt2/tokenizer.json");
    assert(tok != NULL);

    check_roundtrip(tok, "The quick brown fox jumps over the lazy dog.");
    check_roundtrip(tok, "\xe4\xbd\xa0\xe5\xa5\xbd\xef\xbc\x8c\xe4\xb8\x96\xe7\x95\x8c\xef\xbc\x81");
    check_roundtrip(tok, "Hello \xf0\x9f\x91\x8b world \xf0\x9f\x8c\x8d\xe2\x9c\xa8");
    check_roundtrip(tok, "  leading   spaces\tand\ttabs\nnew  lines   ");

    tokenizer_free(tok);
}

/* --- Cycle 4 (H3): bert load sanity ---------------------------------------- */

static void test_bert_load_sanity(void) {
    tokenizer_t *tok = tokenizer_load(MLXD_FIXTURES_DIR "/bert/tokenizer.json");
    assert(tok != NULL);
    assert(tokenizer_type(tok) == TOKENIZER_WORDPIECE);
    assert(tokenizer_vocab_size(tok) == 30522);
    tokenizer_free(tok);
}

/* --- Cycle 5 (H3): bert known-good vectors --------------------------------- */

/* Oracle: HF tokenizers, add_special_tokens=True, so every vector carries the
 * leading 101 [CLS] / trailing 102 [SEP] the C encode wraps unconditionally.
 * ASCII-only inputs by design: encode_wordpiece implements ASCII lowercasing/
 * punctuation only (no Unicode lowercase, accent strip, CJK spacing). */
static void test_bert_known_vectors(void) {
    tokenizer_t *tok = tokenizer_load(MLXD_FIXTURES_DIR "/bert/tokenizer.json");
    assert(tok != NULL);

    static const int32_t fox[] = {101,  1996, 4248,  2829, 4419, 14523,
                                  2058, 1996, 13971, 3899, 102};
    check_vector(tok, "the quick brown fox jumps over the lazy dog", fox, 11);

    static const int32_t subword[] = {101, 14477, 20961, 3468, 19204, 3989, 102};
    check_vector(tok, "unaffable tokenization", subword, 7);

    static const int32_t punct[] = {101, 7592, 1010, 2088, 999, 102};
    check_vector(tok, "hello, world!", punct, 6);

    static const int32_t oov[] = {101,   1053,  13777, 5753, 10179, 29477,
                                  16150, 13109, 5714,  10258, 3286, 102};
    check_vector(tok, "qwertzuiopasd flimflam", oov, 12);

    tokenizer_free(tok);
}

/* decode(encode(text)) must equal the oracle's cleaned string (lowercased,
 * punctuation re-spaced by HF decode cleanup, specials skipped). */
static void check_decode_equals(const tokenizer_t *tok, const char *text, const char *expected) {
    int32_t *ids   = NULL;
    int      count = tokenizer_encode_alloc(tok, text, strlen(text), true, &ids);
    assert(count >= 0);
    char *decoded = tokenizer_decode(tok, ids, count);
    assert(decoded != NULL);
    if (strcmp(decoded, expected) != 0) {
        fprintf(stderr, "decode mismatch for %s\n  expected: %s\n  got:      %s\n", text,
                expected, decoded);
        assert(0 && "decode(encode(text)) mismatch");
    }
    free(decoded);
    free(ids);
}

/* --- Cycle 6 (H3): bert decode round-trip modulo lowercasing --------------- */

static void test_bert_roundtrip(void) {
    tokenizer_t *tok = tokenizer_load(MLXD_FIXTURES_DIR "/bert/tokenizer.json");
    assert(tok != NULL);

    check_decode_equals(tok, "the quick brown fox jumps over the lazy dog",
                        "the quick brown fox jumps over the lazy dog");
    check_decode_equals(tok, "unaffable tokenization", "unaffable tokenization");
    check_decode_equals(tok, "hello, world!", "hello, world!");
    check_decode_equals(tok, "qwertzuiopasd flimflam", "qwertzuiopasd flimflam");

    tokenizer_free(tok);
}

/* --- Cycle 7 (H4): gemma load sanity + load time (skip when absent) --------- */

static void test_gemma(void) {
    const char *path = MLXD_FIXTURES_DIR "/gemma4/tokenizer.json";
    if (access(path, R_OK) != 0) {
        printf("[SKIP] gemma4 fixture absent (dev-only, 32 MB, not committed); "
               "see header of %s for setup\n", __FILE__);
        return;
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    tokenizer_t *tok = tokenizer_load(path);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = (double)(t1.tv_sec - t0.tv_sec) * 1e3 + (double)(t1.tv_nsec - t0.tv_nsec) / 1e6;
    printf("gemma4 tokenizer_load: %.1f ms\n", ms);
    /* Generous hard gate (ASAN/debug builds); the printed number is the
     * informal <1s signal on release builds. */
    assert(ms < 5000.0);

    assert(tok != NULL);
    assert(tokenizer_type(tok) == TOKENIZER_SENTENCEPIECE_BPE);
    assert(tokenizer_vocab_size(tok) == 262144);

    /* Cycle 8 (H4): known-good vectors, oracle with add_special_tokens=False
     * (the C sentencepiece path emits no <bos>). encode_sentencepiece runs
     * bpe_merge over the whole text with no per-word split; that is
     * HF-identical for Gemma because the only vocab tokens with an internal
     * U+2581 are pure whitespace runs (verified against the fixture), so a
     * merge can never span non-space content across a word boundary.
     * Confirmed against the HF oracle on 26 adversarial inputs including
     * multi-space runs, punctuation, CJK, and emoji. */

    static const int32_t multispace[] = {236746, 138, 236763, 139, 236755};
    check_vector(tok, "a  b   c", multispace, 5);

    /* U+070F (0xDC 0x8F) is absent from the vocab: byte fallback -> <0xDC><0x8F>. */
    static const int32_t bytefb[] = {12247, 90104, 236787, 236743, 458, 381, 1345};
    check_vector(tok, "byte fallback: \xdc\x8f end", bytefb, 7);

    static const int32_t cjk[] = {144626, 12811};
    check_vector(tok, "\xe4\xbd\xa0\xe5\xa5\xbd\xe4\xb8\x96\xe7\x95\x8c", cjk, 2);

    static const int32_t fox[] = {818, 3823, 8864, 37423, 38167, 1024, 506, 31770, 4799, 236761};
    check_vector(tok, "The quick brown fox jumps over the lazy dog.", fox, 10);

    /* Cycle 9 (H4): lossless round-trip (decode maps U+2581 -> space and
     * <0xNN> byte-fallback tokens back to raw bytes). */
    check_roundtrip(tok, "a  b   c");
    check_roundtrip(tok, "byte fallback: \xdc\x8f end");
    check_roundtrip(tok, "\xe4\xbd\xa0\xe5\xa5\xbd\xe4\xb8\x96\xe7\x95\x8c");
    check_roundtrip(tok, "The quick brown fox jumps over the lazy dog.");

    tokenizer_free(tok);
}

int main(void) {
    test_type_gpt2();
    test_gpt2_known_vectors();
    test_gpt2_roundtrip();
    test_bert_load_sanity();
    test_bert_known_vectors();
    test_bert_roundtrip();
    test_gemma();
    printf("test_tokenizer_fidelity: all tests passed\n");
    return 0;
}
