#ifndef MLXD_MODEL_TOKENIZER_H
#define MLXD_MODEL_TOKENIZER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct tokenizer tokenizer_t;

typedef enum {
    TOKENIZER_UNKNOWN = -1,
    TOKENIZER_BPE,
    TOKENIZER_WORDPIECE,
    TOKENIZER_SENTENCEPIECE_BPE,
} tokenizer_type_t;

/* Load a tokenizer from a HuggingFace tokenizer.json file.
 * BOS/EOS come from the ordered-name heuristic only; tokenizer_config.json
 * overrides are applied exclusively by tokenizer_load_dir, which is the
 * server path.
 * Returns NULL on error. Caller must free with tokenizer_free. */
tokenizer_t *tokenizer_load(const char *path);

/* Load a tokenizer from a model directory. {dir}/tokenizer.json is preferred;
 * when it is ABSENT the legacy vocab.json + merges.txt pair is synthesized
 * into a byte-level BPE tokenizer. The fallback is existence-based only: a
 * present-but-corrupt tokenizer.json fails the load, it does not fall back.
 * This is the server path: it alone applies {dir}/tokenizer_config.json
 * bos_token/eos_token overrides on top of the ordered-name heuristic.
 * Returns NULL on error. Caller must free with tokenizer_free. */
tokenizer_t *tokenizer_load_dir(const char *dir_path);

/* Load a tokenizer from an in-memory tokenizer.json buffer.
 * Like tokenizer_load, BOS/EOS come from the ordered-name heuristic only;
 * only tokenizer_load_dir applies tokenizer_config.json overrides.
 * Returns NULL on error. Caller must free with tokenizer_free. */
tokenizer_t *tokenizer_load_json(const char *json, size_t len);

void tokenizer_free(tokenizer_t *tok);

/* Encode text[0..len) to token IDs. This is the server-path entry point.
 * parse_special=true splits the input around added tokens (ALL added_tokens
 * entries, special or not) and emits their ids atomically (earliest match
 * wins, ties go to the longest key).
 * parse_special=false encodes the whole text as ordinary content: use it for
 * untrusted user text so a literal "<|im_end|>" cannot inject the control
 * token id. Deviation from HF (matches the Zig reference): the flag disables
 * atomic matching for ALL added tokens, including non-special ones like
 * <think> - HF would still match those. The single-map behavior is
 * injection-safe for untrusted text. WordPiece skips the added-token scan
 * entirely, so parse_special
 * only applies to BPE/SentencePiece; the WordPiece [CLS]/[SEP] wrap is not
 * gated by the flag - it is added by the tokenizer, not parsed from input.
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
/* Classified tokenizer kind (encode/decode dispatch family). NULL returns
 * TOKENIZER_UNKNOWN (-1); every loader success yields a non-NULL tokenizer
 * with a valid type, so callers holding a loaded tokenizer never see it. */
tokenizer_type_t tokenizer_type(const tokenizer_t *tok);
/* Number of vocab entries (HF get_vocab_size convention). For a sparse vocab
 * this can be less than max_id + 1; embedding/output sizing must come from
 * config.json, not from this. */
int tokenizer_vocab_size(const tokenizer_t *tok);
int32_t tokenizer_bos_id(const tokenizer_t *tok);
int32_t tokenizer_eos_id(const tokenizer_t *tok);
/* True when the normalizer prepends the SentencePiece dummy prefix U+2581
 * (normalizer type "Prepend", possibly inside a "Sequence"). Set by the
 * loader; NOT yet consumed by the encode path (future stage). */
bool tokenizer_add_dummy_prefix(const tokenizer_t *tok);

#endif
