#ifndef MLXD_MODEL_TOKENIZER_H
#define MLXD_MODEL_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

typedef struct tokenizer tokenizer_t;

typedef enum {
    TOKENIZER_BPE,
    TOKENIZER_WORDPIECE,
    TOKENIZER_SENTENCEPIECE_BPE,
} tokenizer_type_t;

/* Load a tokenizer from a HuggingFace tokenizer.json file.
 * Returns NULL on error. Caller must free with tokenizer_free. */
tokenizer_t *tokenizer_load(const char *path);

/* Load a tokenizer from an in-memory tokenizer.json buffer.
 * Returns NULL on error. Caller must free with tokenizer_free. */
tokenizer_t *tokenizer_load_json(const char *json, size_t len);

void tokenizer_free(tokenizer_t *tok);

/* Encode text to token IDs.
 * Writes at most max_ids IDs into out_ids.
 * Returns the number of IDs written, or -1 on error. */
int tokenizer_encode(const tokenizer_t *tok, const char *text, int32_t *out_ids, int max_ids);

/* Decode a single token ID to its string representation.
 * Returns a pointer to an internal string (do not free). NULL if ID is invalid. */
const char *tokenizer_decode_token(const tokenizer_t *tok, int32_t id);

/* Decode a sequence of token IDs to text.
 * Returns a heap-allocated string. Caller must free. NULL on error. */
char *tokenizer_decode(const tokenizer_t *tok, const int32_t *ids, int count);

/* Accessors */
/* Number of vocab entries (HF get_vocab_size convention). For a sparse vocab
 * this can be less than max_id + 1; embedding/output sizing must come from
 * config.json, not from this. */
int tokenizer_vocab_size(const tokenizer_t *tok);
int32_t tokenizer_bos_id(const tokenizer_t *tok);
int32_t tokenizer_eos_id(const tokenizer_t *tok);

#endif
