/* GENERATED FILE - DO NOT EDIT.
 * Regenerate with `make unicode-tables` (tools/gen_unicode_tables.py).
 * Unicode Character Database version: 16.0.0 (Python stdlib unicodedata).
 * \p{L}: Lu+Ll+Lt+Lm+Lo -> 677 ranges, 141028 codepoints.
 * \p{M}: Mn+Mc+Me -> 321 ranges, 2501 codepoints.
 * \p{N}: Nd+Nl+No -> 144 ranges, 1911 codepoints.
 */
#ifndef MLXD_TOK_UNICODE_TABLES_H
#define MLXD_TOK_UNICODE_TABLES_H

#include <stdint.h>

typedef struct {
    uint32_t lo;
    uint32_t hi;
} uc_range;

/* \p{L} (Lu+Ll+Lt+Lm+Lo), inclusive ranges, sorted by lo. */
#define UC_LETTER_RANGES_COUNT 677
extern const uc_range uc_letter_ranges[UC_LETTER_RANGES_COUNT];

/* \p{M} (Mn+Mc+Me), inclusive ranges, sorted by lo. */
#define UC_MARK_RANGES_COUNT 321
extern const uc_range uc_mark_ranges[UC_MARK_RANGES_COUNT];

/* \p{N} (Nd+Nl+No), inclusive ranges, sorted by lo. */
#define UC_NUMBER_RANGES_COUNT 144
extern const uc_range uc_number_ranges[UC_NUMBER_RANGES_COUNT];

#endif
