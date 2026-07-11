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

int main(void) {
    test_contraction_t();
    test_contraction_re();
    test_contraction_ll();
    test_contraction_case_insensitive();

    test_space_letters();
    test_word_space_word();

    printf("All tok_pretokenize tests passed.\n");
    return 0;
}
