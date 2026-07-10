#include "model/tok_unicode.h"

#include <assert.h>
#include <stdio.h>

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

static void test_decode_past_end(void) {
    const uint8_t text[] = "A";
    cp_info r = decode_codepoint(text, 1, 1);
    assert(r.cp == 0);
    assert(r.len == 0);
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

    assert(!is_letter(0x00D7));
    assert(!is_letter(0x00F7));
    assert(!is_letter('0'));
    assert(!is_letter(' '));
}

/* --- is_mark ------------------------------------------------------------- */

static void test_is_mark(void) {
    assert(is_mark(0x0300));
    assert(is_mark(0x0489));
    assert(!is_mark('A'));
}

/* --- is_digit ------------------------------------------------------------ */

static void test_is_digit(void) {
    assert(is_digit('0'));
    assert(is_digit('9'));
    assert(!is_digit('a'));
    assert(!is_digit(0x0660));
}

/* --- is_letter_or_mark --------------------------------------------------- */

static void test_is_letter_or_mark(void) {
    assert(is_letter_or_mark('A'));
    assert(is_letter_or_mark(0x0300));
    assert(!is_letter_or_mark('0'));
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
    }
}

int main(void) {
    test_decode_ascii();
    test_decode_2byte();
    test_decode_3byte();
    test_decode_4byte();
    test_decode_invalid_lead();
    test_decode_truncated();
    test_decode_invalid_continuation();
    test_decode_past_end();
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
