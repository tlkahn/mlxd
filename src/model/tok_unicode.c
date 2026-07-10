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

    uint32_t end = pos + byte_len;
    if (end > len)
        end = len;

    for (uint32_t i = pos + 1; i < end; i++) {
        if ((text[i] & 0xC0) != 0x80)
            return (cp_info){lead, 1};
        cp = (cp << 6) | (text[i] & 0x3F);
    }

    if (end - pos < byte_len)
        return (cp_info){lead, 1};

    return (cp_info){cp, byte_len};
}

bool is_letter(uint32_t cp) {
    if (cp >= 'A' && cp <= 'Z') return true;
    if (cp >= 'a' && cp <= 'z') return true;
    if (cp >= 0xC0 && cp <= 0x024F) return cp != 0x00D7 && cp != 0x00F7;
    if (cp >= 0x0370 && cp <= 0x03FF) return true;
    if (cp >= 0x0400 && cp <= 0x04FF) return true;
    if (cp >= 0x0530 && cp <= 0x058F) return true;
    if (cp >= 0x0590 && cp <= 0x05FF) return true;
    if (cp >= 0x0600 && cp <= 0x06FF) return true;
    if (cp >= 0x0780 && cp <= 0x07BF) return true;
    if (cp >= 0x0900 && cp <= 0x097F) return true;
    if (cp >= 0x0980 && cp <= 0x0DFF) return true;
    if (cp >= 0x0E00 && cp <= 0x0E7F) return true;
    if (cp >= 0x0E80 && cp <= 0x0EFF) return true;
    if (cp >= 0x10A0 && cp <= 0x10FF) return true;
    if (cp >= 0x1100 && cp <= 0x11FF) return true;
    if (cp >= 0x3040 && cp <= 0x30FF) return true;
    if (cp >= 0x3400 && cp <= 0x4DBF) return true;
    if (cp >= 0x4E00 && cp <= 0x9FFF) return true;
    if (cp >= 0xAC00 && cp <= 0xD7AF) return true;
    return false;
}

bool is_mark(uint32_t cp) {
    if (cp >= 0x0300 && cp <= 0x036F) return true;
    if (cp >= 0x0483 && cp <= 0x0489) return true;
    if (cp >= 0x0591 && cp <= 0x05BD) return true;
    if (cp >= 0x064B && cp <= 0x065F) return true;
    if (cp == 0x0670) return true;
    if (cp >= 0x06D6 && cp <= 0x06DC) return true;
    if (cp >= 0x0900 && cp <= 0x097F) return true;
    if (cp >= 0x1AB0 && cp <= 0x1AFF) return true;
    if (cp >= 0x1DC0 && cp <= 0x1DFF) return true;
    if (cp >= 0x20D0 && cp <= 0x20FF) return true;
    if (cp >= 0xFE20 && cp <= 0xFE2F) return true;
    return false;
}

bool is_letter_or_mark(uint32_t cp) {
    return is_letter(cp) || is_mark(cp);
}

bool is_digit(uint32_t cp) {
    return cp >= '0' && cp <= '9';
}

bool is_whitespace(uint8_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
           c == 0x0B || c == 0x0C;
}

bool is_whitespace_cp(uint32_t cp) {
    if (cp <= 0xFF)
        return is_whitespace((uint8_t)cp) || cp == 0x00A0;
    return cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200A) ||
           cp == 0x2028 || cp == 0x2029 ||
           cp == 0x202F || cp == 0x205F ||
           cp == 0x3000;
}

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

    for (uint32_t b = 0; b < 256; b++)
        t->cp_to_byte[t->byte_to_cp[b]] = (uint8_t)b;
}
