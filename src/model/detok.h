#ifndef MLXD_MODEL_DETOK_H
#define MLXD_MODEL_DETOK_H

#include <stddef.h>
#include <stdint.h>

typedef struct tokenizer tokenizer_t;
typedef struct detok detok_t;

detok_t *detok_create(const tokenizer_t *tok);

/* Feed one token id. *out is a heap-allocated buffer of newly completed UTF-8
 * bytes (never NULL, NUL-terminated, may be empty). *out_len is its byte
 * length. Withholds incomplete trailing UTF-8 sequences. Caller frees *out.
 * Returns 0 on success, -1 on error. */
int detok_feed(detok_t *d, int32_t id, char **out, size_t *out_len);

/* Emit any withheld trailing bytes verbatim. Same output contract as feed. */
int detok_flush(detok_t *d, char **out, size_t *out_len);

void detok_free(detok_t *d);

#endif
