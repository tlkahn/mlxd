#include "model/tok_unicode.h"
#include "model/tok_unicode_tables.h"

#include <assert.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include <CoreFoundation/CoreFoundation.h>

static char *cf_normalize(const char *input, size_t len,
                          CFStringNormalizationForm form, size_t *out_len) {
    if (len == 0) {
        *out_len = 0;
        return calloc(1, 1);
    }
    CFStringRef src = CFStringCreateWithBytes(
        kCFAllocatorDefault, (const UInt8 *)input, (CFIndex)len,
        kCFStringEncodingUTF8, false);
    if (!src) return NULL;

    CFMutableStringRef mut = CFStringCreateMutableCopy(kCFAllocatorDefault, 0, src);
    CFRelease(src);
    if (!mut) return NULL;

    CFStringNormalize(mut, form);

    CFIndex needed = 0;
    CFStringGetBytes(mut, CFRangeMake(0, CFStringGetLength(mut)),
                     kCFStringEncodingUTF8, 0, false, NULL, 0, &needed);
    char *buf = malloc((size_t)needed + 1);
    if (!buf) { CFRelease(mut); return NULL; }

    CFStringGetBytes(mut, CFRangeMake(0, CFStringGetLength(mut)),
                     kCFStringEncodingUTF8, 0, false, (UInt8 *)buf, needed, NULL);
    buf[needed] = '\0';
    CFRelease(mut);
    *out_len = (size_t)needed;
    return buf;
}

char *uc_normalize_nfc(const char *input, size_t len, size_t *out_len) {
    return cf_normalize(input, len, kCFStringNormalizationFormC, out_len);
}

char *uc_normalize_nfkc(const char *input, size_t len, size_t *out_len) {
    return cf_normalize(input, len, kCFStringNormalizationFormKC, out_len);
}

static bool in_ranges(uint32_t cp, const uc_range *r, size_t n) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (cp < r[mid].lo)
            hi = mid;
        else if (cp > r[mid].hi)
            lo = mid + 1;
        else
            return true;
    }
    return false;
}

/* Invalid UTF-8 decodes as {U+FFFD, 1} so error bytes never classify as
   letters/numbers/whitespace; callers needing the raw byte read text[pos].
   Semantic rejections (overlong, surrogate, > U+10FFFF) also return len=1
   even though the continuation bytes were structurally valid, so iterating
   callers emit one U+FFFD per byte of the bad sequence (per-byte
   replacement, matching the Zig reference; not W3C maximal-subpart). */
uc_cp_info uc_decode_codepoint(const uint8_t *text, uint32_t len, uint32_t pos) {
    if (pos >= len)
        return (uc_cp_info){0, 0};

    uint8_t lead = text[pos];
    uint32_t byte_len;
    uint32_t cp;

    if (lead < 0x80) {
        return (uc_cp_info){lead, 1};
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
        return (uc_cp_info){0xFFFD, 1};
    }

    uint32_t avail = len - pos;
    if (byte_len > avail)
        return (uc_cp_info){0xFFFD, 1};

    for (uint32_t i = 1; i < byte_len; i++) {
        if ((text[pos + i] & 0xC0) != 0x80)
            return (uc_cp_info){0xFFFD, 1};
        cp = (cp << 6) | (text[pos + i] & 0x3F);
    }

    /* Reject overlong, surrogates, and cp > U+10FFFF.
       byte_len is 2-4 here (ASCII returned above), hence the -2 index. */
    static const uint32_t min_cp[] = {0x80, 0x800, 0x10000};
    if (cp < min_cp[byte_len - 2] || (cp >= 0xD800 && cp <= 0xDFFF) || cp > 0x10FFFF)
        return (uc_cp_info){0xFFFD, 1};

    return (uc_cp_info){cp, byte_len};
}

uint32_t uc_encode_codepoint(uint32_t cp, char buf[4]) {
    if (cp < 0x80) {
        buf[0] = (char)cp;
        return 1;
    }
    if (cp < 0x800) {
        buf[0] = (char)(0xC0 | (cp >> 6));
        buf[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp < 0x10000) {
        if (cp >= 0xD800 && cp <= 0xDFFF)
            return 0;
        buf[0] = (char)(0xE0 | (cp >> 12));
        buf[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        buf[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    if (cp <= 0x10FFFF) {
        buf[0] = (char)(0xF0 | (cp >> 18));
        buf[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
        buf[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
        buf[3] = (char)(0x80 | (cp & 0x3F));
        return 4;
    }
    return 0;
}

/* Exact \p{L} via generated UCD range table (see tok_unicode_tables.h). */
bool uc_is_letter(uint32_t cp) {
    if (cp < 0x80)
        return (cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z');
    return in_ranges(cp, uc_letter_ranges, UC_LETTER_RANGES_COUNT);
}

/* Exact \p{M} via generated UCD range table (see tok_unicode_tables.h). */
bool uc_is_mark(uint32_t cp) {
    if (cp < 0x80)
        return false;
    return in_ranges(cp, uc_mark_ranges, UC_MARK_RANGES_COUNT);
}

bool uc_is_letter_or_mark(uint32_t cp) {
    return uc_is_letter(cp) || uc_is_mark(cp);
}

/* Exact \p{N} via generated UCD range table (see tok_unicode_tables.h). */
bool uc_is_number(uint32_t cp) {
    if (cp < 0x80)
        return cp >= '0' && cp <= '9';
    return in_ranges(cp, uc_number_ranges, UC_NUMBER_RANGES_COUNT);
}

bool uc_is_whitespace(uint8_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
           c == 0x0B || c == 0x0C;
}

/* Exact Unicode White_Space property (25 codepoints, stable since Unicode 4.1). */
bool uc_is_whitespace_cp(uint32_t cp) {
    if (cp <= 0xFF)
        return uc_is_whitespace((uint8_t)cp) || cp == 0x0085 || cp == 0x00A0;
    return cp == 0x1680 ||                                       /* Ogham Space Mark */
           (cp >= 0x2000 && cp <= 0x200A) ||                     /* En Quad..Hair Space */
           cp == 0x2028 || cp == 0x2029 ||                       /* Line/Paragraph Separator */
           cp == 0x202F || cp == 0x205F ||                       /* Narrow No-Break / Medium Math */
           cp == 0x3000;                                         /* Ideographic Space */
}

/* 188 printable bytes (94 = '!'-'~', 12 = 0xA1-0xAC, 82 = 0xAE-0xFF) map to themselves.
   The other 68 non-printable bytes map to codepoints 256..323, so max cp is 323.
   UC_BYTES_UNICODE_REV_SIZE must equal 256 + 68 = 324 to hold the full reverse table. */
_Static_assert(UC_BYTES_UNICODE_REV_SIZE == 256 + (256 - 94 - 12 - 82),
               "reverse table must hold 256 + non-printable count");

void uc_build_bytes_to_unicode(uc_bytes_unicode_t *t) {
    /* All-ones bytes read back as UINT16_MAX (= unmapped) in cp_to_byte. */
    memset(t->cp_to_byte, 0xFF, sizeof(t->cp_to_byte));
    uint32_t n = 256;

    for (uint32_t b = 0; b < 256; b++) {
        if ((b >= '!' && b <= '~') ||
            (b >= 0xA1 && b <= 0xAC) ||
            (b >= 0xAE && b <= 0xFF)) {
            t->byte_to_cp[b] = (uint16_t)b;
        } else {
            t->byte_to_cp[b] = (uint16_t)n;
            n++;
        }
        assert(t->byte_to_cp[b] < UC_BYTES_UNICODE_REV_SIZE);
        t->cp_to_byte[t->byte_to_cp[b]] = (uint16_t)b;
    }
}
