#include "model/tokenizer.h"

#include "model/tok_map.h"
#include "model/tok_unicode.h"

#include <yyjson/yyjson.h>

#include <stdlib.h>
#include <string.h>

struct tokenizer {
    tokenizer_type_t   type;
    yyjson_doc        *doc; /* owns all borrowed vocab/merge strings */
    str_u32_map        vocab;
    merge_map          merges;
    const char       **id_to_token;
    uint32_t           id_to_token_cap;
    int                vocab_size;
    int32_t            bos_id;
    int32_t            eos_id;
    uc_bytes_unicode_t bytes_unicode;
};

tokenizer_t *tokenizer_load(const char *path) {
    (void)path;
    return NULL;
}

/* Parse one BPE merge entry, accepting both on-disk formats:
 *   - 2-string array ["left","right"] (newer HF: Qwen3, Gemma 4)
 *   - space-joined string "left right" (GPT-2-style: Qwen2.5, many Llama/Mistral)
 * For the string form, split on the FIRST space: byte-level BPE encodes spaces
 * as 'Ġ', so neither half contains a literal space.
 * Returns false for malformed entries (caller skips them). */
static bool parse_merge_pair(yyjson_val *entry, const char **l, uint32_t *llen,
                             const char **r, uint32_t *rlen) {
    if (yyjson_is_arr(entry) && yyjson_arr_size(entry) >= 2) {
        yyjson_val *lv = yyjson_arr_get(entry, 0);
        yyjson_val *rv = yyjson_arr_get(entry, 1);
        if (!yyjson_is_str(lv) || !yyjson_is_str(rv)) return false;
        *l    = yyjson_get_str(lv);
        *llen = (uint32_t)yyjson_get_len(lv);
        *r    = yyjson_get_str(rv);
        *rlen = (uint32_t)yyjson_get_len(rv);
        return true;
    }
    if (yyjson_is_str(entry)) {
        const char *s  = yyjson_get_str(entry);
        size_t      sl = yyjson_get_len(entry);
        const char *sp = memchr(s, ' ', sl);
        if (!sp) return false;
        *l    = s;
        *llen = (uint32_t)(sp - s);
        *r    = sp + 1;
        *rlen = (uint32_t)(sl - *llen - 1);
        return true;
    }
    return false;
}

tokenizer_t *tokenizer_load_json(const char *json, size_t len) {
    yyjson_doc *doc = yyjson_read(json, len, 0);
    if (!doc) return NULL;

    yyjson_val *root      = yyjson_doc_get_root(doc);
    yyjson_val *model     = yyjson_obj_get(root, "model");
    yyjson_val *vocab_val = yyjson_obj_get(model, "vocab");
    if (!yyjson_is_obj(vocab_val)) {
        yyjson_doc_free(doc);
        return NULL;
    }

    tokenizer_t *tok = calloc(1, sizeof(*tok));
    if (!tok) {
        yyjson_doc_free(doc);
        return NULL;
    }
    tok->doc    = doc;
    tok->type   = TOKENIZER_SENTENCEPIECE_BPE;
    tok->bos_id = -1;
    tok->eos_id = -1;

    /* Vocab: keys are NUL-terminated strings borrowed from the doc. */
    size_t vocab_count = yyjson_obj_size(vocab_val);
    str_u32_map_init(&tok->vocab, vocab_count ? (uint32_t)vocab_count * 2 : 16);

    uint32_t    max_id = 0;
    size_t      idx, max;
    yyjson_val *key, *val;
    yyjson_obj_foreach(vocab_val, idx, max, key, val) {
        uint32_t id = (uint32_t)yyjson_get_int(val);
        str_u32_map_put(&tok->vocab, yyjson_get_str(key), (uint32_t)yyjson_get_len(key), id);
        if (id > max_id) max_id = id;
    }
    tok->vocab_size = (int)tok->vocab.count;

    tok->id_to_token_cap = tok->vocab.count ? max_id + 1 : 0;
    if (tok->id_to_token_cap) {
        tok->id_to_token =
            (const char **)calloc(tok->id_to_token_cap, sizeof(*tok->id_to_token));
        if (!tok->id_to_token) {
            tokenizer_free(tok);
            return NULL;
        }
        yyjson_obj_foreach(vocab_val, idx, max, key, val) {
            tok->id_to_token[(uint32_t)yyjson_get_int(val)] = yyjson_get_str(key);
        }
    }

    yyjson_val *merges_val  = yyjson_obj_get(model, "merges");
    size_t      merge_count = yyjson_is_arr(merges_val) ? yyjson_arr_size(merges_val) : 0;
    merge_map_init(&tok->merges, merge_count ? (uint32_t)merge_count * 2 : 16);
    if (merge_count) {
        yyjson_val *entry;
        yyjson_arr_foreach(merges_val, idx, max, entry) {
            const char *l, *r;
            uint32_t    llen, rlen;
            if (!parse_merge_pair(entry, &l, &llen, &r, &rlen)) continue;
            merge_map_put(&tok->merges, l, llen, r, rlen, (uint32_t)idx);
        }
    }

    return tok;
}

void tokenizer_free(tokenizer_t *tok) {
    if (!tok) return;
    str_u32_map_free(&tok->vocab);
    merge_map_free(&tok->merges);
    free((void *)tok->id_to_token);
    yyjson_doc_free(tok->doc);
    free(tok);
}

int tokenizer_encode(const tokenizer_t *tok, const char *text, int32_t *out_ids, int max_ids) {
    (void)tok;
    (void)text;
    (void)out_ids;
    (void)max_ids;
    return -1;
}

const char *tokenizer_decode_token(const tokenizer_t *tok, int32_t id) {
    if (!tok || id < 0 || (uint32_t)id >= tok->id_to_token_cap) return NULL;
    return tok->id_to_token[id];
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
