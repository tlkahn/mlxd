#include "model/tok_unicode.h"

#include <string.h>

cp_info decode_codepoint(const uint8_t *text, uint32_t len, uint32_t pos) {
    if (pos >= len)
        return (cp_info){0, 0};

    uint8_t lead = text[pos];
    uint32_t byte_len;
    uint32_t cp;

    if (lead < 0x80) {
        return (cp_info){lead, 1};
    } else if ((lead & 0xE0) == 0xC0) {
        byte_len = 2;
        cp = lead & 0x1F;
    } else if ((lead & 0xF0) == 0xE0) {
        byte_len = 3;
        cp = lead & 0x0F;
    } else if ((lead & 0xF8) == 0xF0) {
        byte_len = 4;
        cp = lead & 0x07;
    } else {
        return (cp_info){lead, 1};
    }

    uint32_t avail = len - pos;
    if (byte_len > avail)
        return (cp_info){lead, 1};

    for (uint32_t i = 1; i < byte_len; i++) {
        if ((text[pos + i] & 0xC0) != 0x80)
            return (cp_info){lead, 1};
        cp = (cp << 6) | (text[pos + i] & 0x3F);
    }

    /* Reject overlong, surrogates, and cp > U+10FFFF (matches std.unicode.utf8Decode). */
    static const uint32_t min_cp[] = {0, 0, 0x80, 0x800, 0x10000};
    if (cp < min_cp[byte_len] || (cp >= 0xD800 && cp <= 0xDFFF) || cp > 0x10FFFF)
        return (cp_info){lead, 1};

    return (cp_info){cp, byte_len};
}

/* Approximate \p{L} by major script blocks; extend on false negatives.
   U+00D7 (multiplication sign) and U+00F7 (division sign) are Sm, not letters.
   One line per script keeps parity with the Zig reference (DO NOT merge ranges). */
bool is_letter(uint32_t cp) {
    if (cp >= 'A' && cp <= 'Z') return true;                     /* Basic Latin */
    if (cp >= 'a' && cp <= 'z') return true;                     /* Basic Latin */
    if (cp >= 0xC0 && cp <= 0x024F) return cp != 0x00D7 && cp != 0x00F7; /* Latin Ext A/B */
    if (cp >= 0x0370 && cp <= 0x03FF) return true;               /* Greek */
    if (cp >= 0x0400 && cp <= 0x04FF) return true;               /* Cyrillic */
    if (cp >= 0x0530 && cp <= 0x058F) return true;               /* Armenian */
    if (cp >= 0x0590 && cp <= 0x05FF) return true;               /* Hebrew */
    if (cp >= 0x0600 && cp <= 0x06FF) return true;               /* Arabic */
    if (cp >= 0x0780 && cp <= 0x07BF) return true;               /* Thaana */
    if (cp >= 0x0900 && cp <= 0x097F) return true;               /* Devanagari */
    if (cp >= 0x0980 && cp <= 0x0DFF) return true;               /* Bengali..Sinhala (coarse; sweeps in Indic digits/marks by design) */
    if (cp >= 0x0E00 && cp <= 0x0E7F) return true;               /* Thai */
    if (cp >= 0x0E80 && cp <= 0x0EFF) return true;               /* Lao */
    if (cp >= 0x10A0 && cp <= 0x10FF) return true;               /* Georgian */
    if (cp >= 0x1100 && cp <= 0x11FF) return true;               /* Hangul Jamo */
    if (cp >= 0x1E00 && cp <= 0x1EFF) return true;               /* Latin Extended Additional (Vietnamese) */
    if (cp >= 0x3040 && cp <= 0x30FF) return true;               /* Hiragana + Katakana */
    if (cp >= 0x3400 && cp <= 0x4DBF) return true;               /* CJK Ext A */
    if (cp >= 0x4E00 && cp <= 0x9FFF) return true;               /* CJK Unified */
    if (cp >= 0xAC00 && cp <= 0xD7AF) return true;               /* Korean Syllables */
    return false;
}

/* Approximate \p{M} - combining marks. Not exhaustive; expand on false negatives.
   Overlap with is_letter is harmless for the tokenizer's regex approximation. */
bool is_mark(uint32_t cp) {
    if (cp >= 0x0300 && cp <= 0x036F) return true;               /* Combining Diacriticals */
    if (cp >= 0x0483 && cp <= 0x0489) return true;               /* Cyrillic combining */
    if (cp >= 0x0591 && cp <= 0x05BD) return true;               /* Hebrew combining */
    if (cp >= 0x064B && cp <= 0x065F) return true;               /* Arabic combining */
    if (cp == 0x0670) return true;                               /* Arabic superscript Alef */
    if (cp >= 0x06D6 && cp <= 0x06DC) return true;               /* Arabic small high ligature */
    if (cp >= 0x0900 && cp <= 0x097F) return true;               /* Devanagari (overlap with letters; harmless) */
    if (cp >= 0x1AB0 && cp <= 0x1AFF) return true;               /* Combining Diacriticals Extended */
    if (cp >= 0x1DC0 && cp <= 0x1DFF) return true;               /* Combining Diacriticals Supplement */
    if (cp >= 0x20D0 && cp <= 0x20FF) return true;               /* Combining Diacriticals for Symbols */
    if (cp >= 0xFE20 && cp <= 0xFE2F) return true;               /* Combining Half Marks */
    return false;
}

bool is_letter_or_mark(uint32_t cp) {
    return is_letter(cp) || is_mark(cp);
}

/* ASCII-only \p{N} - deliberate simplification matching the Zig reference. */
bool is_digit(uint32_t cp) {
    return cp >= '0' && cp <= '9';
}

bool is_whitespace(uint8_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
           c == 0x0B || c == 0x0C;
}

/* Approximate \s beyond Latin-1; extend on false negatives. */
bool is_whitespace_cp(uint32_t cp) {
    if (cp <= 0xFF)
        return is_whitespace((uint8_t)cp) || cp == 0x0085 || cp == 0x00A0;
    return cp == 0x1680 ||                                       /* Ogham Space Mark */
           (cp >= 0x2000 && cp <= 0x200A) ||                     /* En Quad..Hair Space */
           cp == 0x2028 || cp == 0x2029 ||                       /* Line/Paragraph Separator */
           cp == 0x202F || cp == 0x205F ||                       /* Narrow No-Break / Medium Math */
           cp == 0x3000;                                         /* Ideographic Space */
}

/* 188 printable bytes (94 = '!'-'~', 12 = 0xA1-0xAC, 82 = 0xAE-0xFF) map to themselves.
   The other 68 non-printable bytes map to codepoints 256..323, so max cp is 323.
   BYTES_UNICODE_REV_SIZE must equal 256 + 68 = 324 to hold the full reverse table. */
_Static_assert(BYTES_UNICODE_REV_SIZE == 256 + (256 - 94 - 12 - 82),
               "reverse table must hold 256 + non-printable count");

void build_bytes_to_unicode(bytes_unicode_t *t) {
    memset(t, 0, sizeof(*t));
    uint32_t n = 256;

    for (uint32_t b = 0; b < 256; b++) {
        uint8_t byte = (uint8_t)b;
        if ((byte >= '!' && byte <= '~') ||
            (byte >= 0xA1 && byte <= 0xAC) ||
            (byte >= 0xAE && byte <= 0xFF)) {
            t->byte_to_cp[b] = b;
        } else {
            t->byte_to_cp[b] = n;
            n++;
        }
    }

    for (uint32_t b = 0; b < 256; b++) {
        t->cp_to_byte[t->byte_to_cp[b]] = (uint8_t)b;
        t->cp_to_byte_valid[t->byte_to_cp[b]] = true;
    }
}
