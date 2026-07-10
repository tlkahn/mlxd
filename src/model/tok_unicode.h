#ifndef MLXD_TOK_UNICODE_H
#define MLXD_TOK_UNICODE_H

#include <stdbool.h>
#include <stdint.h>

typedef struct {
    uint32_t cp;
    uint32_t len;
} cp_info;

cp_info decode_codepoint(const uint8_t *text, uint32_t len, uint32_t pos);

bool is_letter(uint32_t cp);
bool is_mark(uint32_t cp);
bool is_letter_or_mark(uint32_t cp);
bool is_digit(uint32_t cp);
bool is_whitespace(uint8_t c);
bool is_whitespace_cp(uint32_t cp);

#define BYTES_UNICODE_REV_SIZE 324

typedef struct {
    uint32_t byte_to_cp[256];
    uint8_t  cp_to_byte[BYTES_UNICODE_REV_SIZE];
} bytes_unicode_t;

void build_bytes_to_unicode(bytes_unicode_t *t);

#endif
