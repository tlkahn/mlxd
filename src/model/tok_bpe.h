#ifndef MLXD_TOK_BPE_H
#define MLXD_TOK_BPE_H

/* Internal BPE merge machinery, exposed for tests only. The server path
 * reaches it through tokenizer_encode_alloc. */

#include "model/tokenizer.h"

#include <stdbool.h>
#include <stdint.h>

/* One symbol in the BPE merge worklist. Symbols are always contiguous
 * slices input[start..end] - merging adjacent symbols just extends the
 * left node's range and unlinks the right, so no bytes are ever copied. */
typedef struct {
    uint32_t start;
    uint32_t end;
    int32_t  prev;
    int32_t  next;
    uint32_t ver;
} bpe_node;

/* Candidate pair in the merge heap. */
typedef struct {
    uint32_t rank;
    uint32_t left;
    uint32_t right;
    uint32_t lver;
    uint32_t rver;
} bpe_cand;

/* One pre-token: a byte slice input[off..off+len] produced by
 * gpt2_pretokenize. Slices reference the input, never copies. */
typedef struct {
    uint32_t off;
    uint32_t len;
} pretok_slice;

/* Caller-owned reusable scratch buffers; encode_scratch_reserve is the only
 * allocation site so bpe_merge and gpt2_pretokenize never allocate. */
typedef struct {
    bpe_node     *nodes;
    uint32_t      nodes_cap;
    bpe_cand     *heap;
    uint32_t      heap_cap;
    uint32_t      heap_len;
    int32_t      *ids;
    uint32_t      ids_cap;
    pretok_slice *pretoks; /* filled by gpt2_pretokenize */
    uint32_t      pretoks_cap;
    char         *text; /* normalization buffer (byte-to-unicode, U+2581, lowercase) */
    uint32_t      text_cap;
    int32_t      *out; /* id accumulator across words; bpe_merge resets s->ids */
    uint32_t      out_cap;
    char         *cand; /* WordPiece "##" + piece lookup key */
    uint32_t      cand_cap;
} encode_scratch;

void encode_scratch_init(encode_scratch *s);
/* Grow buffers for an input of input_len bytes:
 * nodes_cap >= len, ids_cap >= len, heap_cap >= 3*len, pretoks_cap >= len,
 * text_cap >= len, out_cap >= len, cand_cap >= len + 2.
 * Returns false if input_len exceeds UINT32_MAX / 3 - heap_cap = 3*len is
 * uint32_t arithmetic - or if an allocation fails. On failure the scratch
 * may be PARTIALLY grown (capacities never shrink) and remains usable at
 * its previously reserved size; callers must not call bpe_merge for this
 * input after a failed reserve. */
bool encode_scratch_reserve(encode_scratch *s, size_t input_len);
void encode_scratch_free(encode_scratch *s);

/* Greedy BPE: repeatedly merge the lowest-ranked adjacent pair (leftmost on
 * ties) until no pair has a rank, then emit vocab IDs for the survivors.
 * Never allocates; *out points into s->ids. Returns the id count, or -1
 * (before reading any input) if len exceeds INT32_MAX or the tokenizer is
 * WordPiece, whose greedy longest-match algorithm this does not implement. */
int bpe_merge(const tokenizer_t *tok, encode_scratch *s, const char *input, size_t len,
              int32_t **out);

/* Split input by the GPT-2 pre-tokenizer regex into slices written to
 * s->pretoks. Never allocates; caller must have called
 * encode_scratch_reserve(s, len). Returns the slice count, or -1 if len
 * exceeds INT32_MAX (before reading any input) or the scratch is
 * under-reserved (mid-scan, before the out-of-bounds write). */
int gpt2_pretokenize(encode_scratch *s, const char *input, size_t len);

/* Byte-level BPE encode (GPT-2/Qwen-style): gpt2_pretokenize into words, map
 * each word's bytes through the byte-to-unicode table, bpe_merge per word.
 * Reserves nodes/heap/ids/pretoks/text/out (every buffer but cand) at 2*len.
 * *out points into scratch (valid until the next encode or free). Returns the
 * id count, or -1 on overflow/allocation failure. */
int encode_byte_level(const tokenizer_t *tok, encode_scratch *s, const char *text, size_t len,
                      int32_t **out);

/* WordPiece encode (BERT-style): ASCII-lowercase, split on ASCII whitespace
 * and punctuation, greedy longest-match per word with continuation pieces
 * prefixed by model.continuing_subword_prefix (default "##"); an
 * unmatchable word emits one unk id. Emits bos/eos around the body when
 * set (Stage F relocates that wrap to the public entry point). Reserves
 * text/out/cand (greedy longest-match, so no merge buffers) at
 * len + 2 + prefix_len; *out points into scratch. Returns the id count, or
 * -1 on overflow/allocation failure. */
int encode_wordpiece(const tokenizer_t *tok, encode_scratch *s, const char *text, size_t len,
                     int32_t **out);

/* SentencePiece BPE encode (Gemma-style): replace each ' ' with U+2581 and
 * bpe_merge the whole string - no pre-tokenization. Reserves nodes/heap/ids/
 * text (ids return via bpe_merge, so no pretoks/out/cand) at 3*len; *out
 * points into scratch. Returns the id count, or -1 on overflow/allocation
 * failure. */
int encode_sentencepiece(const tokenizer_t *tok, encode_scratch *s, const char *text,
                         size_t len, int32_t **out);

/* Byte-level BPE decode: map each token's codepoints back through the
 * byte-to-unicode table (unmapped codepoints pass through as raw UTF-8).
 * Unknown ids are skipped. Returns a malloc'd NUL-terminated
 * string ("" for count == 0), or NULL on allocation failure. */
char *decode_byte_level(const tokenizer_t *tok, const int32_t *ids, int count);

/* WordPiece decode: drop registered special tokens, glue continuation
 * pieces (the configured prefix, default "##") without a space, separate
 * other tokens with single spaces. Unknown ids are skipped. Returns
 * a malloc'd NUL-terminated string ("" for count == 0), or NULL on
 * allocation failure. */
char *decode_wordpiece(const tokenizer_t *tok, const int32_t *ids, int count);

/* SentencePiece decode: <0xNN> tokens emit the raw byte, others verbatim;
 * then every U+2581 becomes ' '. strip_leading_space drops exactly one
 * leading space. Unknown ids are skipped. Returns a malloc'd NUL-terminated
 * string ("" for count == 0), or NULL on allocation failure. */
char *decode_sentencepiece(const tokenizer_t *tok, const int32_t *ids, int count,
                           bool strip_leading_space);

#endif
