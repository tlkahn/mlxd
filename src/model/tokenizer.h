#ifndef MLXD_MODEL_TOKENIZER_H
#define MLXD_MODEL_TOKENIZER_H

#include <stdbool.h>
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

/* Encode text[0..len) to token IDs. This is the server-path entry point.
 * parse_special=true splits the input around added special tokens and emits
 * their ids atomically (earliest match wins, ties go to the longest key).
 * parse_special=false encodes the whole text as ordinary content: use it for
 * untrusted user text so a literal "<|im_end|>" cannot inject the control
 * token id. The WordPiece [CLS]/[SEP] wrap is not gated by the flag - it is
 * added by the tokenizer, not parsed from input.
 * Returns the id count, or -1 on error. *out_ids is malloc'd (caller frees)
 * and NULL when the count is 0. */
int tokenizer_encode_alloc(const tokenizer_t *tok, const char *text, size_t len,
                           bool parse_special, int32_t **out_ids);

/* Encode NUL-terminated text to token IDs (fixed buffer, parse_special
 * always true). For tests and simple tools only; the server path must use
 * tokenizer_encode_alloc. Writes at most max_ids IDs into out_ids and
 * returns the FULL id count (snprintf-style): a return > max_ids means the
 * output was truncated. Returns -1 on error. */
int tokenizer_encode(const tokenizer_t *tok, const char *text, int32_t *out_ids, int max_ids);

/* Decode a single token ID to its string representation.
 * Returns a pointer to an internal string (do not free). NULL if ID is invalid. */
const char *tokenizer_decode_token(const tokenizer_t *tok, int32_t id);

/* Decode a sequence of token IDs to text.
 * Special tokens: WordPiece drops registered specials ([CLS], [SEP], ...);
 * byte-level BPE and SentencePiece render them verbatim (e.g. bos/eos as
 * <|endoftext|>), so callers must strip specials from ids themselves. The
 * serving path never feeds them back: the engine signals end-of-stream out
 * of band instead of emitting the eos token.
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
