#include "model/tok_bpe.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

/* reserve -> gpt2_pretokenize -> assert count == n and each slice's bytes
 * (input+off, len) equal want[i]. */
static void expect_pretokens(const char *input, const char *const *want, size_t n) {
    encode_scratch s;
    encode_scratch_init(&s);
    size_t len = strlen(input);
    assert(encode_scratch_reserve(&s, len));
    int count = gpt2_pretokenize(&s, input, len);
    assert(count >= 0);
    assert((size_t)count == n);
    for (size_t i = 0; i < n; i++) {
        size_t wlen = strlen(want[i]);
        assert(s.pretoks[i].len == wlen);
        assert(memcmp(input + s.pretoks[i].off, want[i], wlen) == 0);
    }
    encode_scratch_free(&s);
}

/* --- D1: contractions + bare letter runs ------------------------------------ */

static void test_contraction_t(void) {
    const char *want[] = {"don", "'t"};
    expect_pretokens("don't", want, 2);
}

static void test_contraction_re(void) {
    const char *want[] = {"they", "'re"};
    expect_pretokens("they're", want, 2);
}

static void test_contraction_ll(void) {
    const char *want[] = {"we", "'ll"};
    expect_pretokens("we'll", want, 2);
}

static void test_contraction_case_insensitive(void) {
    const char *want[] = {"DON", "'T"};
    expect_pretokens("DON'T", want, 2);
}

/* --- D2: optional leading non-LNN char -------------------------------------- */

static void test_space_letters(void) {
    const char *want[] = {" total"};
    expect_pretokens(" total", want, 1);
}

static void test_word_space_word(void) {
    const char *want[] = {"def", " total"};
    expect_pretokens("def total", want, 2);
}

/* --- D3: single-number pre-tokens ------------------------------------------- */

static void test_digits_split(void) {
    const char *want[] = {"1", "0", "0"};
    expect_pretokens("100", want, 3);
}

static void test_arabic_indic_digits_split(void) {
    /* \p{N} is one codepoint per slice even when multi-byte (U+0665). */
    const char *want[] = {"\xD9\xA5", "\xD9\xA5"};
    expect_pretokens("\xD9\xA5\xD9\xA5", want, 2);
}

/* --- D4: optional space + punct run + newline tail --------------------------- */

static void test_space_punct(void) {
    const char *want[] = {" ="};
    expect_pretokens(" =", want, 1);
}

static void test_space_punct_run(void) {
    const char *want[] = {" +="};
    expect_pretokens(" +=", want, 1);
}

static void test_space_punct_then_word(void) {
    const char *want[] = {" +=", " foo"};
    expect_pretokens(" += foo", want, 2);
}

static void test_space_star_eq(void) {
    const char *want[] = {" *="};
    expect_pretokens(" *=", want, 1);
}

static void test_punct_newline_tail(void) {
    const char *want[] = {"):\n"};
    expect_pretokens("):\n", want, 1);
}

/* --- D5: whitespace + newline runs ------------------------------------------ */

static void test_word_newline(void) {
    const char *want[] = {"x", "\n"};
    expect_pretokens("x\n", want, 2);
}

static void test_word_spaces_newline(void) {
    const char *want[] = {"x", "   \n"};
    expect_pretokens("x   \n", want, 2);
}

static void test_word_newline_run(void) {
    const char *want[] = {"x", "\n\n"};
    expect_pretokens("x\n\n", want, 2);
}

/* --- D6: trailing-whitespace backtrack + fallback whitespace ------------------ */

static void test_trailing_spaces(void) {
    /* End of input satisfies the (?!\S) lookahead: all whitespace matches. */
    const char *want[] = {"x", "   "};
    expect_pretokens("x   ", want, 2);
}

static void test_indented_assignment(void) {
    /* Pattern 6 leaves the last space of the leading run for pattern 2; the
     * lone " " before "0" falls through to pattern 7 because a 1-cp run
     * followed by \S cannot satisfy the lookahead. */
    const char *want[] = {"   ", " total", " =", " ", "0"};
    expect_pretokens("    total = 0", want, 5);
}

static void test_space_digits(void) {
    const char *want[] = {" ", "1", "0", "0"};
    expect_pretokens(" 100", want, 4);
}

int main(void) {
    test_contraction_t();
    test_contraction_re();
    test_contraction_ll();
    test_contraction_case_insensitive();

    test_space_letters();
    test_word_space_word();

    test_digits_split();
    test_arabic_indic_digits_split();

    test_space_punct();
    test_space_punct_run();
    test_space_punct_then_word();
    test_space_star_eq();
    test_punct_newline_tail();

    test_word_newline();
    test_word_spaces_newline();
    test_word_newline_run();

    test_trailing_spaces();
    test_indented_assignment();
    test_space_digits();

    printf("All tok_pretokenize tests passed.\n");
    return 0;
}
