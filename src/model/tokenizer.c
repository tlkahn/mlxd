#include "model/tokenizer.h"

#include "model/tok_bpe.h"
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

/* Check if a pre_tokenizer JSON value contains a ByteLevel type, recursing
 * into Sequence.pretokenizers. */
static bool has_byte_level(yyjson_val *pt) {
    if (!yyjson_is_obj(pt)) return false;
    yyjson_val *t = yyjson_obj_get(pt, "type");
    if (!yyjson_is_str(t)) return false;
    if (strcmp(yyjson_get_str(t), "ByteLevel") == 0) return true;
    if (strcmp(yyjson_get_str(t), "Sequence") == 0) {
        yyjson_val *pts = yyjson_obj_get(pt, "pretokenizers");
        if (yyjson_is_arr(pts)) {
            size_t      idx, max;
            yyjson_val *sub;
            yyjson_arr_foreach(pts, idx, max, sub) {
                if (has_byte_level(sub)) return true;
            }
        }
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
    tok->bos_id = -1;
    tok->eos_id = -1;
    uc_build_bytes_to_unicode(&tok->bytes_unicode);

    yyjson_val *model_type = yyjson_obj_get(model, "type");
    if (yyjson_is_str(model_type) && strcmp(yyjson_get_str(model_type), "WordPiece") == 0)
        tok->type = TOKENIZER_WORDPIECE;
    else if (has_byte_level(yyjson_obj_get(root, "pre_tokenizer")))
        tok->type = TOKENIZER_BPE;
    else
        tok->type = TOKENIZER_SENTENCEPIECE_BPE;

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

/* --- BPE merge (heap + linked list) ---------------------------------------- */

/* Encode a codepoint <= 0x7FF as UTF-8; the GPT-2 byte table's max mapped
 * codepoint is 323, so two bytes always suffice. */
static uint32_t utf8_encode_cp(uint32_t cp, char buf[4]) {
    if (cp < 0x80) {
        buf[0] = (char)cp;
        return 1;
    }
    buf[0] = (char)(0xC0 | (cp >> 6));
    buf[1] = (char)(0x80 | (cp & 0x3F));
    return 2;
}

void encode_scratch_init(encode_scratch *s) { memset(s, 0, sizeof(*s)); }

void encode_scratch_reserve(encode_scratch *s, size_t input_len) {
    uint32_t len = input_len ? (uint32_t)input_len : 1;
    if (s->nodes_cap < len) {
        s->nodes     = realloc(s->nodes, len * sizeof(*s->nodes));
        s->nodes_cap = len;
    }
    if (s->ids_cap < len) {
        s->ids     = realloc(s->ids, len * sizeof(*s->ids));
        s->ids_cap = len;
    }
    if (s->heap_cap < 3 * len) {
        s->heap     = realloc(s->heap, 3 * len * sizeof(*s->heap));
        s->heap_cap = 3 * len;
    }
}

void encode_scratch_free(encode_scratch *s) {
    free(s->nodes);
    free(s->heap);
    free(s->ids);
    memset(s, 0, sizeof(*s));
}

/* Heap order (rank asc, left node index asc): node indices are creation order
 * = text order and merges never create nodes, so the secondary key reproduces
 * the naive scan's leftmost-wins tie-break exactly. */
static bool bpe_cand_less(const bpe_cand *a, const bpe_cand *b) {
    if (a->rank != b->rank) return a->rank < b->rank;
    return a->left < b->left;
}

/* Push (l,r) as a candidate if the pair's current slices have a merge rank. */
static void bpe_push_cand(const tokenizer_t *tok, encode_scratch *s, const char *input,
                          uint32_t l, uint32_t r) {
    const bpe_node *ln = &s->nodes[l];
    const bpe_node *rn = &s->nodes[r];
    uint32_t        rank;
    if (!merge_map_get(&tok->merges, input + ln->start, ln->end - ln->start, input + rn->start,
                       rn->end - rn->start, &rank))
        return;
    s->heap[s->heap_len++] = (bpe_cand){rank, l, r, ln->ver, rn->ver};
    uint32_t i             = s->heap_len - 1;
    while (i > 0) {
        uint32_t parent = (i - 1) / 2;
        if (!bpe_cand_less(&s->heap[i], &s->heap[parent])) break;
        bpe_cand tmp    = s->heap[parent];
        s->heap[parent] = s->heap[i];
        s->heap[i]      = tmp;
        i               = parent;
    }
}

static bpe_cand bpe_heap_pop(encode_scratch *s) {
    bpe_cand top = s->heap[0];
    s->heap[0]   = s->heap[--s->heap_len];
    uint32_t i   = 0;
    for (;;) {
        uint32_t smallest = i;
        uint32_t lc       = 2 * i + 1;
        uint32_t rc       = 2 * i + 2;
        if (lc < s->heap_len && bpe_cand_less(&s->heap[lc], &s->heap[smallest])) smallest = lc;
        if (rc < s->heap_len && bpe_cand_less(&s->heap[rc], &s->heap[smallest])) smallest = rc;
        if (smallest == i) break;
        bpe_cand tmp      = s->heap[smallest];
        s->heap[smallest] = s->heap[i];
        s->heap[i]        = tmp;
        i                 = smallest;
    }
    return top;
}

int bpe_merge(const tokenizer_t *tok, encode_scratch *s, const char *input, size_t len,
              int32_t **out) {
    /* Split into individual UTF-8 characters. */
    uint32_t n_nodes = 0;
    uint32_t pos     = 0;
    while (pos < len) {
        uc_cp_info info = uc_decode_codepoint((const uint8_t *)input, (uint32_t)len, pos);
        uint32_t   end  = pos + info.len;
        s->nodes[n_nodes] = (bpe_node){
            .start = pos,
            .end   = end,
            .prev  = (int32_t)n_nodes - 1,
            .next  = end < len ? (int32_t)n_nodes + 1 : -1,
            .ver   = 0,
        };
        n_nodes++;
        pos = end;
    }

    /* Seed the heap with every ranked adjacent pair. */
    s->heap_len = 0;
    for (uint32_t j = 0; j + 1 < n_nodes; j++) bpe_push_cand(tok, s, input, j, j + 1);

    while (s->heap_len > 0) {
        bpe_cand  c  = bpe_heap_pop(s);
        bpe_node *ns = s->nodes;

        /* Stale-candidate guard: any merge touching a node bumps its version,
         * so a snapshot mismatch means the candidate references content that
         * changed since the push - either an unlinked node or one whose slice
         * grew (e.g. seeded (a,b) after (b,c) merged). */
        if (ns[c.left].ver != c.lver || ns[c.right].ver != c.rver) continue;

        /* Merge right into left: extend range, unlink right, invalidate every
         * candidate that referenced either node's old content. */
        ns[c.left].end = ns[c.right].end;
        ns[c.left].ver++;
        ns[c.right].ver++;
        ns[c.left].next = ns[c.right].next;
        if (ns[c.right].next >= 0) ns[ns[c.right].next].prev = (int32_t)c.left;

        if (ns[c.left].prev >= 0) bpe_push_cand(tok, s, input, (uint32_t)ns[c.left].prev, c.left);
        if (ns[c.left].next >= 0) bpe_push_cand(tok, s, input, c.left, (uint32_t)ns[c.left].next);
    }

    /* Map surviving symbols to vocab IDs. */
    int count = 0;
    for (int32_t cur = n_nodes > 0 ? 0 : -1; cur >= 0; cur = s->nodes[cur].next) {
        const bpe_node *n = &s->nodes[cur];
        uint32_t        id;
        if (str_u32_map_get(&tok->vocab, input + n->start, n->end - n->start, &id)) {
            s->ids[count++] = (int32_t)id;
        } else if (tok->type == TOKENIZER_BPE) {
            /* Byte-level BPE: every mapped byte is a base vocab token in any
             * well-formed vocab, so a miss is pathological; skip it. */
            for (uint32_t b = n->start; b < n->end; b++) {
                char     buf[4];
                uint32_t blen =
                    utf8_encode_cp(tok->bytes_unicode.byte_to_cp[(uint8_t)input[b]], buf);
                if (str_u32_map_get(&tok->vocab, buf, blen, &id)) s->ids[count++] = (int32_t)id;
            }
        }
    }
    *out = s->ids;
    return count;
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
