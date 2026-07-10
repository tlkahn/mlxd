#include "model/tok_unicode.h"

#include <assert.h>
#include <stdio.h>
#include <sys/mman.h>
#ifndef MAP_ANON
#define MAP_ANON 0x1000
#endif

/* --- decode_codepoint ---------------------------------------------------- */

static void test_decode_ascii(void) {
    const uint8_t text[] = "A";
    cp_info r = decode_codepoint(text, 1, 0);
    assert(r.cp == 0x41);
    assert(r.len == 1);
}

static void test_decode_2byte(void) {
    const uint8_t text[] = "\xC3\xA9";
    cp_info r = decode_codepoint(text, 2, 0);
    assert(r.cp == 0xE9);
    assert(r.len == 2);
}

static void test_decode_3byte(void) {
    const uint8_t text[] = "\xE2\x9C\x93";
    cp_info r = decode_codepoint(text, 3, 0);
    assert(r.cp == 0x2713);
    assert(r.len == 3);
}

static void test_decode_4byte(void) {
    const uint8_t text[] = "\xF0\x9F\x98\x80";
    cp_info r = decode_codepoint(text, 4, 0);
    assert(r.cp == 0x1F600);
    assert(r.len == 4);
}

static void test_decode_invalid_lead(void) {
    const uint8_t text[] = "\xFF";
    cp_info r = decode_codepoint(text, 1, 0);
    assert(r.cp == 0xFF);
    assert(r.len == 1);
}

static void test_decode_truncated(void) {
    const uint8_t text[] = "\xC3";
    cp_info r = decode_codepoint(text, 1, 0);
    assert(r.cp == 0xC3);
    assert(r.len == 1);
}

static void test_decode_invalid_continuation(void) {
    const uint8_t text[] = "\xC3\x00";
    cp_info r = decode_codepoint(text, 2, 0);
    assert(r.cp == 0xC3);
    assert(r.len == 1);
}

static void test_decode_overlong(void) {
    /* 2-byte overlong for U+0000 */
    const uint8_t ol2[] = "\xC0\x80";
    cp_info r = decode_codepoint(ol2, 2, 0);
    assert(r.cp == 0xC0 && r.len == 1);

    /* 3-byte overlong for U+0000 */
    const uint8_t ol3[] = "\xE0\x80\x80";
    r = decode_codepoint(ol3, 3, 0);
    assert(r.cp == 0xE0 && r.len == 1);

    /* 4-byte overlong for U+0000 */
    const uint8_t ol4[] = "\xF0\x80\x80\x80";
    r = decode_codepoint(ol4, 4, 0);
    assert(r.cp == 0xF0 && r.len == 1);
}

static void test_decode_surrogate(void) {
    /* U+D800 (low surrogate start) */
    const uint8_t lo[] = "\xED\xA0\x80";
    cp_info r = decode_codepoint(lo, 3, 0);
    assert(r.cp == 0xED && r.len == 1);

    /* U+DFFF (high surrogate end) */
    const uint8_t hi[] = "\xED\xBF\xBF";
    r = decode_codepoint(hi, 3, 0);
    assert(r.cp == 0xED && r.len == 1);
}

static void test_decode_out_of_range(void) {
    /* U+110000 - one past the max valid codepoint */
    const uint8_t a[] = "\xF4\x90\x80\x80";
    cp_info r = decode_codepoint(a, 4, 0);
    assert(r.cp == 0xF4 && r.len == 1);

    /* U+1FFFFF - max encodable by 4-byte form */
    const uint8_t b[] = "\xF7\xBF\xBF\xBF";
    r = decode_codepoint(b, 4, 0);
    assert(r.cp == 0xF7 && r.len == 1);
}

static void test_decode_boundaries(void) {
    /* Minimal valid encodings at each byte-length boundary */
    const uint8_t u0080[] = "\xC2\x80";
    cp_info r = decode_codepoint(u0080, 2, 0);
    assert(r.cp == 0x0080 && r.len == 2);

    const uint8_t u0800[] = "\xE0\xA0\x80";
    r = decode_codepoint(u0800, 3, 0);
    assert(r.cp == 0x0800 && r.len == 3);

    const uint8_t u10000[] = "\xF0\x90\x80\x80";
    r = decode_codepoint(u10000, 4, 0);
    assert(r.cp == 0x10000 && r.len == 4);

    /* Max valid codepoint U+10FFFF */
    const uint8_t u10ffff[] = "\xF4\x8F\xBF\xBF";
    r = decode_codepoint(u10ffff, 4, 0);
    assert(r.cp == 0x10FFFF && r.len == 4);

    /* Surrogate neighbors: U+D7FF and U+E000 must be accepted */
    const uint8_t ud7ff[] = "\xED\x9F\xBF";
    r = decode_codepoint(ud7ff, 3, 0);
    assert(r.cp == 0xD7FF && r.len == 3);

    const uint8_t ue000[] = "\xEE\x80\x80";
    r = decode_codepoint(ue000, 3, 0);
    assert(r.cp == 0xE000 && r.len == 3);
}

static void test_decode_past_end(void) {
    const uint8_t text[] = "A";
    cp_info r = decode_codepoint(text, 1, 1);
    assert(r.cp == 0);
    assert(r.len == 0);
}

static void test_decode_pos_overflow(void) {
    size_t sz = (size_t)UINT32_MAX + 1;
    uint8_t *map = mmap(NULL, sz, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANON, -1, 0);
    if (map == MAP_FAILED) {
        printf("  (skipping pos-overflow test: mmap failed)\n");
        return;
    }
    /* Place a 4-byte lead at pos UINT32_MAX - 1 with only 2 bytes available.
       pos + byte_len = 0xFFFFFFFE + 4 = wraps to 2 in uint32_t,
       so the old code would not detect truncation. */
    uint32_t pos = UINT32_MAX - 1;
    map[pos] = 0xF0;
    map[pos + 1] = 0x90;

    cp_info r = decode_codepoint(map, UINT32_MAX, pos);
    assert(r.cp == 0xF0 && r.len == 1);

    munmap(map, sz);
}

/* --- is_letter ----------------------------------------------------------- */

static void test_is_letter(void) {
    assert(is_letter('A'));
    assert(is_letter('z'));
    assert(is_letter(0x05D0));
    assert(is_letter(0x0E01));
    assert(is_letter(0x0995));
    assert(is_letter(0x0531));
    assert(is_letter(0x10A0));
    assert(is_letter(0x4E00));
    assert(is_letter(0x3042));

    assert(is_letter(0x1E00));
    assert(is_letter(0x1EA0));
    assert(is_letter(0x1EFF));

    assert(!is_letter(0x00D7));
    assert(!is_letter(0x00F7));
    assert(!is_letter('0'));
    assert(!is_letter(' '));

    /* Exact \p{L}: non-letters inside old coarse blocks must be rejected */
    assert(!is_letter(0x09E6));  /* BENGALI DIGIT ZERO (Nd) */
    assert(!is_letter(0x0964));  /* DEVANAGARI DANDA (Po) */
    assert(!is_letter(0x093E));  /* DEVANAGARI VOWEL SIGN AA (Mc) */

    /* Exact \p{L}: letters outside old coarse blocks must be accepted */
    assert(is_letter(0x0250));   /* LATIN SMALL LETTER TURNED A (Ll, IPA) */
    assert(is_letter(0x02B0));   /* MODIFIER LETTER SMALL H (Lm) */
    assert(is_letter(0x10400));  /* DESERET CAPITAL LETTER LONG I (Lu, astral) */

    assert(!is_letter(0x1F600)); /* emoji (So) */
}

/* --- is_mark ------------------------------------------------------------- */

static void test_is_mark(void) {
    assert(is_mark(0x0300));
    assert(is_mark(0x0489));
    assert(!is_mark('A'));

    /* Exact \p{M}: Devanagari non-marks inside the old block claim */
    assert(!is_mark(0x0915));    /* DEVANAGARI LETTER KA (Lo) */
    assert(!is_mark(0x0966));    /* DEVANAGARI DIGIT ZERO (Nd) */
    assert(!is_mark(0x0964));    /* DEVANAGARI DANDA (Po) */

    /* Exact \p{M}: real Devanagari marks stay in */
    assert(is_mark(0x0901));     /* SIGN CANDRABINDU (Mn) */
    assert(is_mark(0x093E));     /* VOWEL SIGN AA (Mc) */
}

/* --- is_digit ------------------------------------------------------------ */

static void test_is_digit(void) {
    assert(is_digit('0'));
    assert(is_digit('9'));
    assert(!is_digit('a'));

    /* Exact \p{N}: non-ASCII numbers count */
    assert(is_digit(0x0660));    /* ARABIC-INDIC DIGIT ZERO (Nd) */
    assert(is_digit(0x09E6));    /* BENGALI DIGIT ZERO (Nd) */
    assert(is_digit(0x0966));    /* DEVANAGARI DIGIT ZERO (Nd) */
    assert(is_digit(0x2160));    /* ROMAN NUMERAL ONE (Nl) */
    assert(is_digit(0x00BD));    /* VULGAR FRACTION ONE HALF (No) */

    assert(!is_digit(0x0965));   /* DEVANAGARI DOUBLE DANDA (Po) */
}

/* --- is_letter_or_mark --------------------------------------------------- */

static void test_is_letter_or_mark(void) {
    assert(is_letter_or_mark('A'));
    assert(is_letter_or_mark(0x0300));
    assert(!is_letter_or_mark('0'));

    /* Exact L|M union: letters and marks in, digits and punctuation out */
    assert(is_letter_or_mark(0x0915));   /* DEVANAGARI LETTER KA (Lo) */
    assert(is_letter_or_mark(0x093E));   /* DEVANAGARI VOWEL SIGN AA (Mc) */
    assert(!is_letter_or_mark(0x0964));  /* DEVANAGARI DANDA (Po) */
    assert(!is_letter_or_mark(0x09E6));  /* BENGALI DIGIT ZERO (Nd) */
}

/* --- is_whitespace (byte) ------------------------------------------------ */

static void test_is_whitespace(void) {
    assert(is_whitespace(' '));
    assert(is_whitespace('\t'));
    assert(is_whitespace('\n'));
    assert(is_whitespace('\r'));
    assert(is_whitespace(0x0B));
    assert(is_whitespace(0x0C));
    assert(!is_whitespace('a'));
    assert(!is_whitespace(0));
}

/* --- is_whitespace_cp (codepoint) ---------------------------------------- */

static void test_is_whitespace_cp(void) {
    assert(is_whitespace_cp(' '));
    assert(is_whitespace_cp(0x0085));
    assert(!is_whitespace_cp(0x0086));
    assert(is_whitespace_cp(0x00A0));
    assert(is_whitespace_cp(0x2003));
    assert(is_whitespace_cp(0x2009));
    assert(is_whitespace_cp(0x2028));
    assert(is_whitespace_cp(0x2029));
    assert(is_whitespace_cp(0x3000));
    assert(!is_whitespace_cp('a'));
}

/* --- build_bytes_to_unicode ---------------------------------------------- */

static void test_bytes_to_unicode_identity(void) {
    bytes_unicode_t t;
    build_bytes_to_unicode(&t);

    assert(t.byte_to_cp['!'] == '!');
    assert(t.byte_to_cp['~'] == '~');
    assert(t.byte_to_cp[0xA1] == 0xA1);
    assert(t.byte_to_cp[0xFF] == 0xFF);
}

static void test_bytes_to_unicode_offset(void) {
    bytes_unicode_t t;
    build_bytes_to_unicode(&t);

    assert(t.byte_to_cp[0x00] == 0x100);
    assert(t.byte_to_cp['\n'] == 0x10A);
    assert(t.byte_to_cp[' '] == 0x120);
    assert(t.byte_to_cp[0xAD] >= 0x100);
}

static void test_bytes_to_unicode_roundtrip(void) {
    bytes_unicode_t t;
    build_bytes_to_unicode(&t);

    for (uint32_t b = 0; b < 256; b++) {
        uint32_t cp = t.byte_to_cp[b];
        assert(cp < BYTES_UNICODE_REV_SIZE);
        assert(t.cp_to_byte[cp] == (uint8_t)b);
        assert(t.cp_to_byte_valid[cp]);
    }

    assert(!t.cp_to_byte_valid[0x00]);
    assert(!t.cp_to_byte_valid[0x20]);
    assert(!t.cp_to_byte_valid[0x7F]);
    assert(!t.cp_to_byte_valid[0xAD]);
}

int main(void) {
    test_decode_ascii();
    test_decode_2byte();
    test_decode_3byte();
    test_decode_4byte();
    test_decode_invalid_lead();
    test_decode_truncated();
    test_decode_invalid_continuation();
    test_decode_overlong();
    test_decode_surrogate();
    test_decode_out_of_range();
    test_decode_boundaries();
    test_decode_past_end();
    test_decode_pos_overflow();
    test_is_letter();
    test_is_mark();
    test_is_digit();
    test_is_letter_or_mark();
    test_is_whitespace();
    test_is_whitespace_cp();
    test_bytes_to_unicode_identity();
    test_bytes_to_unicode_offset();
    test_bytes_to_unicode_roundtrip();
    printf("All tok_unicode tests passed.\n");
    return 0;
}
