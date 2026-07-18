/* -D_POSIX_C_SOURCE hides MAP_ANON on Darwin; re-expose the OS extensions
   instead of hardcoding the platform-specific flag value. */
#define _DARWIN_C_SOURCE

#include "model/tok_unicode.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

/* --- uc_decode_codepoint ---------------------------------------------------- */

static void test_decode_ascii(void) {
    const uint8_t text[] = "A";
    uc_cp_info r = uc_decode_codepoint(text, 1, 0);
    assert(r.cp == 0x41);
    assert(r.len == 1);
}

static void test_decode_2byte(void) {
    const uint8_t text[] = "\xC3\xA9";
    uc_cp_info r = uc_decode_codepoint(text, 2, 0);
    assert(r.cp == 0xE9);
    assert(r.len == 2);
}

static void test_decode_3byte(void) {
    const uint8_t text[] = "\xE2\x9C\x93";
    uc_cp_info r = uc_decode_codepoint(text, 3, 0);
    assert(r.cp == 0x2713);
    assert(r.len == 3);
}

static void test_decode_4byte(void) {
    const uint8_t text[] = "\xF0\x9F\x98\x80";
    uc_cp_info r = uc_decode_codepoint(text, 4, 0);
    assert(r.cp == 0x1F600);
    assert(r.len == 4);
}

static void test_decode_invalid_lead(void) {
    const uint8_t text[] = "\xFF";
    uc_cp_info r = uc_decode_codepoint(text, 1, 0);
    assert(r.cp == 0xFFFD);
    assert(r.len == 1);
}

static void test_decode_truncated(void) {
    const uint8_t text[] = "\xC3";
    uc_cp_info r = uc_decode_codepoint(text, 1, 0);
    assert(r.cp == 0xFFFD);
    assert(r.len == 1);
}

static void test_decode_invalid_continuation(void) {
    const uint8_t text[] = "\xC3\x00";
    uc_cp_info r = uc_decode_codepoint(text, 2, 0);
    assert(r.cp == 0xFFFD);
    assert(r.len == 1);
}

static void test_decode_overlong(void) {
    /* 2-byte overlong for U+0000 */
    const uint8_t ol2[] = "\xC0\x80";
    uc_cp_info r = uc_decode_codepoint(ol2, 2, 0);
    assert(r.cp == 0xFFFD && r.len == 1);

    /* 3-byte overlong for U+0000 */
    const uint8_t ol3[] = "\xE0\x80\x80";
    r = uc_decode_codepoint(ol3, 3, 0);
    assert(r.cp == 0xFFFD && r.len == 1);

    /* 4-byte overlong for U+0000 */
    const uint8_t ol4[] = "\xF0\x80\x80\x80";
    r = uc_decode_codepoint(ol4, 4, 0);
    assert(r.cp == 0xFFFD && r.len == 1);
}

static void test_decode_surrogate(void) {
    /* U+D800 (high surrogate range start, D800-DBFF) */
    const uint8_t hs[] = "\xED\xA0\x80";
    uc_cp_info r = uc_decode_codepoint(hs, 3, 0);
    assert(r.cp == 0xFFFD && r.len == 1);

    /* U+DFFF (low/trail surrogate range end, DC00-DFFF) */
    const uint8_t ls[] = "\xED\xBF\xBF";
    r = uc_decode_codepoint(ls, 3, 0);
    assert(r.cp == 0xFFFD && r.len == 1);
}

static void test_decode_semantic_reject_per_byte(void) {
    /* Semantic rejections (surrogates, overlongs) return len=1, so a caller
       iterating with pos += r.len emits one U+FFFD per byte of the bad
       sequence (per-byte replacement, matching the Zig reference). */
    const uint8_t hs[] = "\xED\xA0\x80"; /* U+D800 high surrogate */
    uint32_t count = 0;
    for (uint32_t pos = 0; pos < 3;) {
        uc_cp_info r = uc_decode_codepoint(hs, 3, pos);
        assert(r.cp == 0xFFFD && r.len == 1);
        pos += r.len;
        count++;
    }
    assert(count == 3);

    const uint8_t ol3[] = "\xE0\x80\x80"; /* 3-byte overlong for U+0000 */
    count = 0;
    for (uint32_t pos = 0; pos < 3;) {
        uc_cp_info r = uc_decode_codepoint(ol3, 3, pos);
        assert(r.cp == 0xFFFD && r.len == 1);
        pos += r.len;
        count++;
    }
    assert(count == 3);
}

static void test_decode_out_of_range(void) {
    /* U+110000 - one past the max valid codepoint */
    const uint8_t a[] = "\xF4\x90\x80\x80";
    uc_cp_info r = uc_decode_codepoint(a, 4, 0);
    assert(r.cp == 0xFFFD && r.len == 1);

    /* U+1FFFFF - max encodable by 4-byte form */
    const uint8_t b[] = "\xF7\xBF\xBF\xBF";
    r = uc_decode_codepoint(b, 4, 0);
    assert(r.cp == 0xFFFD && r.len == 1);
}

static void test_decode_error_not_letter(void) {
    /* Regression: a truncated 0xC0 lead must not surface as U+00C0 (A-grave),
       which is a letter; error cps must never classify as letters. */
    const uint8_t text[] = "\xC0\x20";
    uc_cp_info r = uc_decode_codepoint(text, 2, 0);
    assert(r.cp == 0xFFFD && r.len == 1);
    assert(!uc_is_letter(r.cp));
    assert(!uc_is_number(r.cp));
    assert(!uc_is_whitespace_cp(r.cp));
}

static void test_decode_boundaries(void) {
    /* Minimal valid encodings at each byte-length boundary */
    const uint8_t u0080[] = "\xC2\x80";
    uc_cp_info r = uc_decode_codepoint(u0080, 2, 0);
    assert(r.cp == 0x0080 && r.len == 2);

    const uint8_t u0800[] = "\xE0\xA0\x80";
    r = uc_decode_codepoint(u0800, 3, 0);
    assert(r.cp == 0x0800 && r.len == 3);

    const uint8_t u10000[] = "\xF0\x90\x80\x80";
    r = uc_decode_codepoint(u10000, 4, 0);
    assert(r.cp == 0x10000 && r.len == 4);

    /* Max valid codepoint U+10FFFF */
    const uint8_t u10ffff[] = "\xF4\x8F\xBF\xBF";
    r = uc_decode_codepoint(u10ffff, 4, 0);
    assert(r.cp == 0x10FFFF && r.len == 4);

    /* Surrogate neighbors: U+D7FF and U+E000 must be accepted */
    const uint8_t ud7ff[] = "\xED\x9F\xBF";
    r = uc_decode_codepoint(ud7ff, 3, 0);
    assert(r.cp == 0xD7FF && r.len == 3);

    const uint8_t ue000[] = "\xEE\x80\x80";
    r = uc_decode_codepoint(ue000, 3, 0);
    assert(r.cp == 0xE000 && r.len == 3);
}

static void test_decode_past_end(void) {
    const uint8_t text[] = "A";
    uc_cp_info r = uc_decode_codepoint(text, 1, 1);
    assert(r.cp == 0);
    assert(r.len == 0);
}

static void test_decode_pos_overflow(void) {
    size_t sz = (size_t)UINT32_MAX + 1;
    uint8_t *map = mmap(NULL, sz, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANON, -1, 0);
    /* Hard-fail rather than skip: make test redirects stdout/stderr to
       /dev/null, so a skip message would silently drop this coverage. */
    assert(map != MAP_FAILED);
    /* Place a 4-byte lead at pos UINT32_MAX - 1 with only 2 bytes available.
       pos + byte_len = 0xFFFFFFFE + 4 = wraps to 2 in uint32_t,
       so the old code would not detect truncation. */
    uint32_t pos = UINT32_MAX - 1;
    map[pos] = 0xF0;
    map[pos + 1] = 0x90;

    uc_cp_info r = uc_decode_codepoint(map, UINT32_MAX, pos);
    assert(r.cp == 0xFFFD && r.len == 1);

    munmap(map, sz);
}

/* --- uc_is_letter ----------------------------------------------------------- */

static void test_uc_is_letter(void) {
    assert(uc_is_letter('A'));
    assert(uc_is_letter('z'));
    assert(uc_is_letter(0x05D0));
    assert(uc_is_letter(0x0E01));
    assert(uc_is_letter(0x0995));
    assert(uc_is_letter(0x0531));
    assert(uc_is_letter(0x10A0));
    assert(uc_is_letter(0x4E00));
    assert(uc_is_letter(0x3042));

    assert(uc_is_letter(0x1E00));
    assert(uc_is_letter(0x1EA0));
    assert(uc_is_letter(0x1EFF));

    assert(!uc_is_letter(0x00D7));
    assert(!uc_is_letter(0x00F7));
    assert(!uc_is_letter('0'));
    assert(!uc_is_letter(' '));

    /* Exact \p{L}: non-letters inside old coarse blocks must be rejected */
    assert(!uc_is_letter(0x09E6));  /* BENGALI DIGIT ZERO (Nd) */
    assert(!uc_is_letter(0x0964));  /* DEVANAGARI DANDA (Po) */
    assert(!uc_is_letter(0x093E));  /* DEVANAGARI VOWEL SIGN AA (Mc) */

    /* Exact \p{L}: letters outside old coarse blocks must be accepted */
    assert(uc_is_letter(0x0250));   /* LATIN SMALL LETTER TURNED A (Ll, IPA) */
    assert(uc_is_letter(0x02B0));   /* MODIFIER LETTER SMALL H (Lm) */
    assert(uc_is_letter(0x10400));  /* DESERET CAPITAL LETTER LONG I (Lu, astral) */

    assert(!uc_is_letter(0x1F600)); /* emoji (So) */
}

/* --- uc_is_mark ------------------------------------------------------------- */

static void test_uc_is_mark(void) {
    assert(uc_is_mark(0x0300));
    assert(uc_is_mark(0x0489));
    assert(!uc_is_mark('A'));

    /* Exact \p{M}: Devanagari non-marks inside the old block claim */
    assert(!uc_is_mark(0x0915));    /* DEVANAGARI LETTER KA (Lo) */
    assert(!uc_is_mark(0x0966));    /* DEVANAGARI DIGIT ZERO (Nd) */
    assert(!uc_is_mark(0x0964));    /* DEVANAGARI DANDA (Po) */

    /* Exact \p{M}: real Devanagari marks stay in */
    assert(uc_is_mark(0x0901));     /* SIGN CANDRABINDU (Mn) */
    assert(uc_is_mark(0x093E));     /* VOWEL SIGN AA (Mc) */
}

/* --- uc_is_number ------------------------------------------------------------ */

static void test_uc_is_number(void) {
    assert(uc_is_number('0'));
    assert(uc_is_number('9'));
    assert(!uc_is_number('a'));

    /* Exact \p{N}: non-ASCII numbers count */
    assert(uc_is_number(0x0660));    /* ARABIC-INDIC DIGIT ZERO (Nd) */
    assert(uc_is_number(0x09E6));    /* BENGALI DIGIT ZERO (Nd) */
    assert(uc_is_number(0x0966));    /* DEVANAGARI DIGIT ZERO (Nd) */
    assert(uc_is_number(0x2160));    /* ROMAN NUMERAL ONE (Nl) */
    assert(uc_is_number(0x00BD));    /* VULGAR FRACTION ONE HALF (No) */

    assert(!uc_is_number(0x0965));   /* DEVANAGARI DOUBLE DANDA (Po) */
}

/* --- uc_is_letter_or_mark --------------------------------------------------- */

static void test_uc_is_letter_or_mark(void) {
    assert(uc_is_letter_or_mark('A'));
    assert(uc_is_letter_or_mark(0x0300));
    assert(!uc_is_letter_or_mark('0'));

    /* Exact L|M union: letters and marks in, digits and punctuation out */
    assert(uc_is_letter_or_mark(0x0915));   /* DEVANAGARI LETTER KA (Lo) */
    assert(uc_is_letter_or_mark(0x093E));   /* DEVANAGARI VOWEL SIGN AA (Mc) */
    assert(!uc_is_letter_or_mark(0x0964));  /* DEVANAGARI DANDA (Po) */
    assert(!uc_is_letter_or_mark(0x09E6));  /* BENGALI DIGIT ZERO (Nd) */
}

/* --- uc_is_whitespace (byte) ------------------------------------------------ */

static void test_uc_is_whitespace(void) {
    assert(uc_is_whitespace(' '));
    assert(uc_is_whitespace('\t'));
    assert(uc_is_whitespace('\n'));
    assert(uc_is_whitespace('\r'));
    assert(uc_is_whitespace(0x0B));
    assert(uc_is_whitespace(0x0C));
    assert(!uc_is_whitespace('a'));
    assert(!uc_is_whitespace(0));
}

/* --- uc_is_whitespace_cp (codepoint) ---------------------------------------- */

static void test_uc_is_whitespace_cp(void) {
    assert(uc_is_whitespace_cp(' '));
    assert(uc_is_whitespace_cp(0x0085));
    assert(!uc_is_whitespace_cp(0x0086));
    assert(uc_is_whitespace_cp(0x00A0));
    assert(uc_is_whitespace_cp(0x2003));
    assert(uc_is_whitespace_cp(0x2009));
    assert(uc_is_whitespace_cp(0x2028));
    assert(uc_is_whitespace_cp(0x2029));
    assert(uc_is_whitespace_cp(0x3000));
    assert(!uc_is_whitespace_cp('a'));
}

static void test_uc_is_whitespace_cp_exact(void) {
    /* uc_is_whitespace_cp covers exactly the 25 codepoints with the Unicode
       White_Space property (stable since Unicode 4.1). */
    static const uint32_t ws[] = {
        0x0009, 0x000A, 0x000B, 0x000C, 0x000D, 0x0020, 0x0085, 0x00A0,
        0x1680, 0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006,
        0x2007, 0x2008, 0x2009, 0x200A, 0x2028, 0x2029, 0x202F, 0x205F,
        0x3000,
    };
    size_t n = sizeof(ws) / sizeof(ws[0]);
    assert(n == 25);

    size_t next = 0;
    for (uint32_t cp = 0; cp <= 0x10FFFF; cp++) {
        bool expect = next < n && cp == ws[next];
        assert(uc_is_whitespace_cp(cp) == expect);
        if (expect)
            next++;
    }
    assert(next == n);
}

/* --- uc_encode_codepoint ----------------------------------------------------- */

static void test_encode_codepoint(void) {
    char buf[4];
    assert(uc_encode_codepoint(0x41, buf) == 1 && buf[0] == 'A');
    assert(uc_encode_codepoint(0xE9, buf) == 2 && memcmp(buf, "\xC3\xA9", 2) == 0);
    assert(uc_encode_codepoint(0x20AC, buf) == 3 && memcmp(buf, "\xE2\x82\xAC", 3) == 0);
    assert(uc_encode_codepoint(0x1F600, buf) == 4 && memcmp(buf, "\xF0\x9F\x98\x80", 4) == 0);

    /* Round-trip every length-class boundary through uc_decode_codepoint. */
    static const uint32_t cps[] = {0x00,   0x7F,   0x80,    0x7FF,    0x800,
                                   0xD7FF, 0xE000, 0xFFFF,  0x10000,  0x10FFFF};
    for (size_t i = 0; i < sizeof(cps) / sizeof(cps[0]); i++) {
        uint32_t n = uc_encode_codepoint(cps[i], buf);
        assert(n >= 1 && n <= 4);
        uc_cp_info r = uc_decode_codepoint((const uint8_t *)buf, n, 0);
        assert(r.cp == cps[i]);
        assert(r.len == n);
    }
}

static void test_encode_codepoint_rejects(void) {
    char buf[4];
    assert(uc_encode_codepoint(0xD800, buf) == 0);   /* surrogate low bound */
    assert(uc_encode_codepoint(0xDFFF, buf) == 0);   /* surrogate high bound */
    assert(uc_encode_codepoint(0x110000, buf) == 0); /* past Unicode max */
    assert(uc_encode_codepoint(UINT32_MAX, buf) == 0);
}

/* --- uc_utf8_scan ------------------------------------------------------------ */

static void test_utf8_scan_ascii(void) {
    assert(uc_utf8_scan("hello", 5) == UC_SCAN_ASCII);
}

static void test_utf8_scan_empty(void) {
    assert(uc_utf8_scan("", 0) == UC_SCAN_ASCII);
}

static void test_utf8_scan_valid_multibyte(void) {
    /* e-acute: \xC3\xA9 */
    assert(uc_utf8_scan("\xc3\xa9", 2) == UC_SCAN_VALID);
}

static void test_utf8_scan_invalid_lead(void) {
    assert(uc_utf8_scan("\xff", 1) == UC_SCAN_INVALID);
}

static void test_utf8_scan_truncated(void) {
    /* 2-byte lead with no continuation */
    assert(uc_utf8_scan("\xc3", 1) == UC_SCAN_INVALID);
}

static void test_utf8_scan_overlong(void) {
    /* 2-byte overlong for U+0000 */
    assert(uc_utf8_scan("\xc0\x80", 2) == UC_SCAN_INVALID);
}

static void test_utf8_scan_surrogate(void) {
    /* U+D800 encoded as 3 bytes */
    assert(uc_utf8_scan("\xed\xa0\x80", 3) == UC_SCAN_INVALID);
}

static void test_utf8_scan_literal_fffd(void) {
    /* Real U+FFFD is valid UTF-8 (3 bytes: EF BF BD) */
    assert(uc_utf8_scan("\xef\xbf\xbd", 3) == UC_SCAN_VALID);
}

static void test_utf8_scan_mixed_ascii_multibyte(void) {
    /* "cafe\xcc\x81" = "cafe" + combining acute */
    assert(uc_utf8_scan("cafe\xcc\x81", 6) == UC_SCAN_VALID);
}

static void test_utf8_scan_invalid_in_middle(void) {
    /* valid prefix + invalid byte + valid suffix */
    assert(uc_utf8_scan("a\xff" "b", 3) == UC_SCAN_INVALID);
}

/* --- uc_normalize_nfc -------------------------------------------------------- */

static void test_normalize_nfc_compose(void) {
    size_t n;
    /* e + combining acute -> precomposed e-acute */
    char *r = uc_normalize_nfc("e\xcc\x81", 3, &n);
    assert(r != NULL);
    assert(n == 2);
    assert(memcmp(r, "\xc3\xa9", 2) == 0);
    assert(r[n] == '\0');
    free(r);

    /* ASCII unchanged */
    r = uc_normalize_nfc("hello", 5, &n);
    assert(r != NULL);
    assert(n == 5);
    assert(memcmp(r, "hello", 5) == 0);
    free(r);

    /* Already-composed is idempotent */
    r = uc_normalize_nfc("\xc3\xa9", 2, &n);
    assert(r != NULL);
    assert(n == 2);
    assert(memcmp(r, "\xc3\xa9", 2) == 0);
    free(r);
}

static void test_normalize_empty(void) {
    size_t n = 999;
    char *r = uc_normalize_nfc("", 0, &n);
    assert(r != NULL);
    assert(n == 0);
    free(r);
}

static void test_normalize_invalid_utf8(void) {
    size_t n = 777;
    char *r = uc_normalize_nfc("\xff\xfe", 2, &n);
    assert(r == NULL);
    assert(n == 0);
}

/* --- uc_normalize_nfkc ------------------------------------------------------- */

static void test_normalize_nfkc_compat(void) {
    size_t n;
    /* U+FB01 LATIN SMALL LIGATURE FI -> "fi" under NFKC */
    char *r = uc_normalize_nfkc("\xef\xac\x81", 3, &n);
    assert(r != NULL);
    assert(n == 2);
    assert(memcmp(r, "fi", 2) == 0);
    free(r);

    /* NFC leaves the ligature intact */
    r = uc_normalize_nfc("\xef\xac\x81", 3, &n);
    assert(r != NULL);
    assert(n == 3);
    assert(memcmp(r, "\xef\xac\x81", 3) == 0);
    free(r);

    /* NFKC also does canonical composition */
    r = uc_normalize_nfkc("e\xcc\x81", 3, &n);
    assert(r != NULL);
    assert(n == 2);
    assert(memcmp(r, "\xc3\xa9", 2) == 0);
    free(r);
}

/* --- uc_build_bytes_to_unicode ---------------------------------------------- */

static void test_bytes_to_unicode_identity(void) {
    uc_bytes_unicode_t t;
    uc_build_bytes_to_unicode(&t);

    assert(t.byte_to_cp['!'] == '!');
    assert(t.byte_to_cp['~'] == '~');
    assert(t.byte_to_cp[0xA1] == 0xA1);
    assert(t.byte_to_cp[0xFF] == 0xFF);
}

static void test_bytes_to_unicode_offset(void) {
    uc_bytes_unicode_t t;
    uc_build_bytes_to_unicode(&t);

    assert(t.byte_to_cp[0x00] == 0x100);
    assert(t.byte_to_cp['\n'] == 0x10A);
    assert(t.byte_to_cp[' '] == 0x120);
    assert(t.byte_to_cp[0xAD] >= 0x100);
}

static void test_bytes_to_unicode_roundtrip(void) {
    uc_bytes_unicode_t t;
    uc_build_bytes_to_unicode(&t);

    for (uint32_t b = 0; b < 256; b++) {
        uint32_t cp = t.byte_to_cp[b];
        assert(cp < UC_BYTES_UNICODE_REV_SIZE);
        assert(t.cp_to_byte[cp] != UINT16_MAX);
        assert(t.cp_to_byte[cp] == b);
    }

    /* Non-printable byte values have no reverse mapping at their own cp. */
    assert(t.cp_to_byte[0x00] == UINT16_MAX);
    assert(t.cp_to_byte[0x20] == UINT16_MAX);
    assert(t.cp_to_byte[0x7F] == UINT16_MAX);
    assert(t.cp_to_byte[0xAD] == UINT16_MAX);
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
    test_decode_semantic_reject_per_byte();
    test_decode_out_of_range();
    test_decode_error_not_letter();
    test_decode_boundaries();
    test_decode_past_end();
    test_decode_pos_overflow();
    test_uc_is_letter();
    test_uc_is_mark();
    test_uc_is_number();
    test_uc_is_letter_or_mark();
    test_uc_is_whitespace();
    test_uc_is_whitespace_cp();
    test_uc_is_whitespace_cp_exact();
    test_encode_codepoint();
    test_encode_codepoint_rejects();
    test_utf8_scan_ascii();
    test_utf8_scan_empty();
    test_utf8_scan_valid_multibyte();
    test_utf8_scan_invalid_lead();
    test_utf8_scan_truncated();
    test_utf8_scan_overlong();
    test_utf8_scan_surrogate();
    test_utf8_scan_literal_fffd();
    test_utf8_scan_mixed_ascii_multibyte();
    test_utf8_scan_invalid_in_middle();
    test_normalize_nfc_compose();
    test_normalize_empty();
    test_normalize_invalid_utf8();
    test_normalize_nfkc_compat();
    test_bytes_to_unicode_identity();
    test_bytes_to_unicode_offset();
    test_bytes_to_unicode_roundtrip();
    printf("All tok_unicode tests passed.\n");
    return 0;
}
