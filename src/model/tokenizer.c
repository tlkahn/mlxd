#include "model/tokenizer.h"

#include <stdlib.h>

/* TODO: full BPE/WordPiece/SentencePiece implementation during model module migration */

struct tokenizer {
    tokenizer_type_t type;
    int              vocab_size;
    int32_t          bos_id;
    int32_t          eos_id;
    /* vocab, merges, pre-tokenizer state, etc. */
};

tokenizer_t *tokenizer_load(const char *path) {
    (void)path;
    return NULL;
}

void tokenizer_free(tokenizer_t *tok) { free(tok); }

int tokenizer_encode(const tokenizer_t *tok, const char *text, int32_t *out_ids, int max_ids) {
    (void)tok;
    (void)text;
    (void)out_ids;
    (void)max_ids;
    return -1;
}

const char *tokenizer_decode_token(const tokenizer_t *tok, int32_t id) {
    (void)tok;
    (void)id;
    return NULL;
}

char *tokenizer_decode(const tokenizer_t *tok, const int32_t *ids, int count) {
    (void)tok;
    (void)ids;
    (void)count;
    return NULL;
}

int tokenizer_vocab_size(const tokenizer_t *tok) { return tok ? tok->vocab_size : 0; }

int32_t tokenizer_bos_id(const tokenizer_t *tok) { return tok ? tok->bos_id : -1; }

int32_t tokenizer_eos_id(const tokenizer_t *tok) { return tok ? tok->eos_id : -1; }
