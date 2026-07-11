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

/* --- D7: full Python snippet (integration pin) -------------------------------
 * Expected split is the HF `tokenizers` reference output. */

static void test_python_snippet(void) {
    const char *want[] = {"def", " total", "(items", "):\n", "   ", " total", " =", " ", "0"};
    expect_pretokens("def total(items):\n    total = 0", want, 9);
}

/* --- D8: punct/letter interleaving (pattern ordering pin) --------------------- */

static void test_special_token_shape(void) {
    /* Pattern 2's optional non-LNN char beats pattern 4 for "_start". */
    const char *want[] = {"<|", "im", "_start", "|>"};
    expect_pretokens("<|im_start|>", want, 4);
}

/* --- D9: non-Latin letter runs ------------------------------------------------ */

static void test_hebrew_run(void) {
    const char *want[] = {"\xD7\xA9\xD7\x9C\xD7\x95\xD7\x9D"};
    expect_pretokens("\xD7\xA9\xD7\x9C\xD7\x95\xD7\x9D", want, 1); /* שלום */
}

static void test_space_hebrew_run(void) {
    const char *want[] = {" \xD7\xA9\xD7\x9C\xD7\x95\xD7\x9D"};
    expect_pretokens(" \xD7\xA9\xD7\x9C\xD7\x95\xD7\x9D", want, 1);
}

static void test_thai_run(void) {
    const char *want[] = {"\xE0\xB9\x84\xE0\xB8\x97\xE0\xB8\xA2"};
    expect_pretokens("\xE0\xB9\x84\xE0\xB8\x97\xE0\xB8\xA2", want, 1); /* ไทย */
}

static void test_hebrew_then_punct(void) {
    const char *want[] = {"\xD7\xA9\xD7\x9C\xD7\x95\xD7\x9D", "."};
    expect_pretokens("\xD7\xA9\xD7\x9C\xD7\x95\xD7\x9D.", want, 2);
}

static void test_thai_then_punct(void) {
    const char *want[] = {"\xE0\xB9\x84\xE0\xB8\x97\xE0\xB8\xA2", "!"};
    expect_pretokens("\xE0\xB9\x84\xE0\xB8\x97\xE0\xB8\xA2!", want, 2);
}

/* --- D10: exact-Unicode pins (beyond the Zig reference) ------------------------
 * Pins the exact \s decision (Arabic-Indic \p{N} is pinned in D3): NBSP
 * (U+00A0, 2 bytes) is whitespace, so pattern 6 backtracks one 2-byte
 * codepoint and pattern 2 absorbs the last NBSP as its optional char. A
 * byte-level \s regression would split the NBSPs mid-sequence. */

static void test_nbsp_backtrack(void) {
    /* \142 is 'b' - octal, so it can follow a hex escape without splicing. */
    const char *want[] = {"a", "\xC2\xA0", "\xC2\xA0\142"};
    expect_pretokens("a\xC2\xA0\xC2\xA0\142", want, 3);
}

/* --- review-fix characterization pins -------------------------------------------
 * Lock behavior that the PR-28 review refactors could disturb: uppercase
 * 2-letter contraction suffixes (case-fold change), leading/bare combining
 * marks (letter-run restructure), NBSP-then-\S fallthrough (single-decode
 * change), and whitespace-only inputs (pattern 6+7 merge). */

static void test_contraction_upper_2char(void) {
    const char *want[] = {"THEY", "'RE"};
    expect_pretokens("THEY'RE", want, 2);
}

static void test_nbsp_then_punct(void) {
    /* NBSP followed by \S falls through patterns 2-6 and lands in pattern 7
     * as a single-codepoint whitespace pre-token; the '.' follows via 4. */
    const char *want[] = {"\xC2\xA0", "."};
    expect_pretokens("\xC2\xA0.", want, 2);
}

static void test_leading_combining_mark(void) {
    /* U+0301 is \p{M}: pattern 2's optional non-LNN char absorbs it, then
     * the letter run continues through 'x'. */
    const char *want[] = {"\xCC\x81x"};
    expect_pretokens("\xCC\x81x", want, 1);
}

static void test_bare_combining_mark(void) {
    /* U+0301 alone matches pattern 2 with the optional char empty: a mark
     * is a valid [\p{L}\p{M}]+ run start. */
    const char *want[] = {"\xCC\x81"};
    expect_pretokens("\xCC\x81", want, 1);
}

static void test_ws_only_single(void) {
    const char *want[] = {" "};
    expect_pretokens(" ", want, 1);
}

static void test_ws_only_nbsp(void) {
    const char *want[] = {"\xC2\xA0"};
    expect_pretokens("\xC2\xA0", want, 1);
}

/* --- D5 fix: \s* prefix may contain newlines -----------------------------------
 * `\s*[\r\n]+` backtracks: the greedy \s* consumes newlines and later
 * whitespace, and the match ends one past the LAST \r/\n byte of the run.
 * Expected splits verified against the HF reference regex. */

static void test_cr_space_lf(void) {
    const char *want[] = {"\r \n"};
    expect_pretokens("\r \n", want, 1);
}

static void test_lf_tab_lf(void) {
    const char *want[] = {"\n\t\n"};
    expect_pretokens("\n\t\n", want, 1);
}

static void test_nl_ws_nl_then_word(void) {
    const char *want[] = {"\n \n", " x"};
    expect_pretokens("\n \n x", want, 2);
}

static void test_word_cr_sp_lf_word(void) {
    const char *want[] = {"a", "\r \n", "b"};
    expect_pretokens("a\r \nb", want, 3);
}

static void test_nl_then_spaces_word(void) {
    /* Over-match guard: nothing after the '\n' is part of pattern 5; the
     * two spaces resolve via the pattern-6 backtrack then pattern 2. */
    const char *want[] = {"\n", " ", " x"};
    expect_pretokens("\n  x", want, 3);
}

/* --- scratch reserve contract ---------------------------------------------------- */

static void test_pretoks_cap_guard(void) {
    /* An under-reserved scratch must fail with -1, not write out of
     * bounds - the reserve contract has no assert in release builds. */
    encode_scratch s;
    encode_scratch_init(&s);
    assert(encode_scratch_reserve(&s, 1));        /* cap = 1 */
    assert(gpt2_pretokenize(&s, "100", 3) == -1); /* needs 3 slices */
    encode_scratch_free(&s);
}

/* --- edge cases ---------------------------------------------------------------- */

static void test_empty_input(void) {
    expect_pretokens("", NULL, 0);
}

static void test_lone_apostrophe_at_end(void) {
    /* Pattern 1's i+1 < len guard fails; the quote is punct via pattern 4. */
    const char *want[] = {"don", "'"};
    expect_pretokens("don'", want, 2);
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

    test_python_snippet();

    test_special_token_shape();

    test_hebrew_run();
    test_space_hebrew_run();
    test_thai_run();
    test_hebrew_then_punct();
    test_thai_then_punct();

    test_nbsp_backtrack();

    test_contraction_upper_2char();
    test_nbsp_then_punct();
    test_leading_combining_mark();
    test_bare_combining_mark();
    test_ws_only_single();
    test_ws_only_nbsp();

    test_cr_space_lf();
    test_lf_tab_lf();
    test_nl_ws_nl_then_word();
    test_word_cr_sp_lf_word();
    test_nl_then_spaces_word();

    test_pretoks_cap_guard();

    test_empty_input();
    test_lone_apostrophe_at_end();

    printf("All tok_pretokenize tests passed.\n");
    return 0;
}
