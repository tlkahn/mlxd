#ifndef MLXD_TOK_UNICODE_H
#define MLXD_TOK_UNICODE_H

#include <stdbool.h>
#include <stdint.h>

typedef struct {
    uint32_t cp;
    uint32_t len;
} uc_cp_info;

uc_cp_info uc_decode_codepoint(const uint8_t *text, uint32_t len, uint32_t pos);

/* Exact Unicode \p{L}, \p{M}, \p{N} classifiers, backed by range tables
   generated from the UCD (tok_unicode_tables.h, `make unicode-tables`). */
bool uc_is_letter(uint32_t cp);
bool uc_is_mark(uint32_t cp);
bool uc_is_letter_or_mark(uint32_t cp);
/* All of \p{N} (Nd+Nl+No): includes Roman numerals and fractions,
   not just decimal digits - do not use for digit-value arithmetic. */
bool uc_is_number(uint32_t cp);
bool uc_is_whitespace(uint8_t c);
bool uc_is_whitespace_cp(uint32_t cp);

/* 188 printable bytes map to themselves; the 68 non-printable bytes map to
   codepoints 256..323, so the reverse table needs 324 entries. */
#define UC_BYTES_UNICODE_REV_SIZE 324

typedef struct {
    /* Max mapped codepoint is 323, so uint16_t suffices. */
    uint16_t byte_to_cp[256];
    /* UINT16_MAX = no byte maps to this codepoint. */
    uint16_t cp_to_byte[UC_BYTES_UNICODE_REV_SIZE];
} uc_bytes_unicode_t;

void uc_build_bytes_to_unicode(uc_bytes_unicode_t *t);

#endif
