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

/* Caller-owned reusable scratch buffers; encode_scratch_reserve is the only
 * allocation site so bpe_merge itself never allocates. */
typedef struct {
    bpe_node *nodes;
    uint32_t  nodes_cap;
    bpe_cand *heap;
    uint32_t  heap_cap;
    uint32_t  heap_len;
    int32_t  *ids;
    uint32_t  ids_cap;
} encode_scratch;

void encode_scratch_init(encode_scratch *s);
/* Grow buffers for an input of input_len bytes:
 * nodes_cap >= len, ids_cap >= len, heap_cap >= 3*len.
 * Returns false if input_len exceeds UINT32_MAX / 3 - heap_cap = 3*len is
 * uint32_t arithmetic - or if an allocation fails. */
bool encode_scratch_reserve(encode_scratch *s, size_t input_len);
void encode_scratch_free(encode_scratch *s);

/* Greedy BPE: repeatedly merge the lowest-ranked adjacent pair (leftmost on
 * ties) until no pair has a rank, then emit vocab IDs for the survivors.
 * Never allocates; *out points into s->ids. Returns the id count, or -1
 * (before reading any input) if len exceeds INT32_MAX or the tokenizer is
 * WordPiece, whose greedy longest-match algorithm this does not implement. */
int bpe_merge(const tokenizer_t *tok, encode_scratch *s, const char *input, size_t len,
              int32_t **out);

#endif
