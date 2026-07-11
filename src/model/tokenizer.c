#include "model/tokenizer.h"

#include "model/tok_bpe.h"
#include "model/tok_map.h"
#include "model/tok_unicode.h"

#include <yyjson/yyjson.h>

#include <assert.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Largest accepted vocab id (4M-1). Real vocabs top out around 262K; anything
 * larger is a malformed or hostile file, not a bigger model. */
#define MLXD_TOK_MAX_ID ((1u << 22) - 1)

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
    int32_t            unk_id; /* vocab id of "<unk>", -1 if absent */
    /* Per-byte fallback token id, resolved once at load (-1 = no vocab entry):
     * BPE maps bytes through the GPT-2 byte-to-unicode table, SentencePiece
     * through <0xNN> tokens, WordPiece has no byte fallback. */
    int32_t            byte_fallback_ids[256];
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
    tok->unk_id = -1;

    /* Initialize both maps up front so every early error path below frees
     * fully-initialized maps instead of relying on the calloc-zeroed struct. */
    size_t      vocab_count = yyjson_obj_size(vocab_val);
    yyjson_val *merges_val  = yyjson_obj_get(model, "merges");
    size_t      merge_count = yyjson_is_arr(merges_val) ? yyjson_arr_size(merges_val) : 0;
    /* Compute the capacity hints in size_t and clamp before the narrowing
     * cast: counts >= 2^31 would wrap the uint32_t doubling. The hint is
     * advisory (maps grow on demand), so clamping is transparent. */
    size_t vocab_hint = vocab_count ? vocab_count * 2 : 16;
    size_t merge_hint = merge_count ? merge_count * 2 : 16;
    if (vocab_hint > (1u << 30)) vocab_hint = 1u << 30;
    if (merge_hint > (1u << 30)) merge_hint = 1u << 30;
    bool vocab_ok  = str_u32_map_init(&tok->vocab, (uint32_t)vocab_hint);
    bool merges_ok = merge_map_init(&tok->merges, (uint32_t)merge_hint);
    if (!vocab_ok || !merges_ok) {
        tokenizer_free(tok);
        return NULL;
    }

    /* Absent model.type is fine (BPE/SP detection below); a present value
     * must be a recognized string - non-string or unrecognized types (e.g.
     * Unigram) would encode through the BPE paths with silently wrong ids. */
    yyjson_val *model_type = yyjson_obj_get(model, "type");
    const char *type_str   = model_type ? yyjson_get_str(model_type) : NULL;
    if (model_type && !type_str) {
        tokenizer_free(tok);
        return NULL;
    }
    if (type_str && strcmp(type_str, "WordPiece") == 0) {
        tok->type = TOKENIZER_WORDPIECE;
    } else if (type_str && strcmp(type_str, "BPE") != 0) {
        tokenizer_free(tok);
        return NULL;
    } else if (has_byte_level(yyjson_obj_get(root, "pre_tokenizer"))) {
        tok->type = TOKENIZER_BPE;
    } else {
        tok->type = TOKENIZER_SENTENCEPIECE_BPE;
    }

    /* Vocab: keys are NUL-terminated strings borrowed from the doc. */
    uint32_t    max_id = 0;
    size_t      idx, max;
    yyjson_val *key, *val;
    yyjson_obj_foreach(vocab_val, idx, max, key, val) {
        /* Ids size id_to_token (calloc of max_id+1), so an unvalidated id is
         * an unbounded allocation; fail the load like the Zig reference. */
        if (!yyjson_is_int(val)) {
            tokenizer_free(tok);
            return NULL;
        }
        int64_t raw = yyjson_get_sint(val);
        if (raw < 0 || raw > MLXD_TOK_MAX_ID) {
            tokenizer_free(tok);
            return NULL;
        }
        uint32_t id = (uint32_t)raw;
        if (!str_u32_map_put(&tok->vocab, yyjson_get_str(key), (uint32_t)yyjson_get_len(key),
                             id)) {
            tokenizer_free(tok);
            return NULL;
        }
        if (id > max_id) max_id = id;
    }
    tok->vocab_size = (int)tok->vocab.count;

    uint32_t unk;
    if (str_u32_map_get(&tok->vocab, "<unk>", 5, &unk)) tok->unk_id = (int32_t)unk;

    /* Resolve the per-byte fallback ids once, now that the vocab map is
     * built; bpe_emit_symbol then indexes instead of re-hashing per byte.
     * The byte-to-unicode table is only needed here, so it lives on the
     * stack (decode rebuilds its own when that stage lands). */
    for (int b = 0; b < 256; b++) tok->byte_fallback_ids[b] = -1;
    if (tok->type == TOKENIZER_BPE) {
        uc_bytes_unicode_t bu;
        uc_build_bytes_to_unicode(&bu);
        for (int b = 0; b < 256; b++) {
            char buf[4];
            /* The byte table's max mapped codepoint is 323, so this always
             * encodes to 1-2 bytes and never fails. */
            uint32_t blen = uc_encode_codepoint(bu.byte_to_cp[b], buf);
            uint32_t id;
            if (str_u32_map_get(&tok->vocab, buf, blen, &id))
                tok->byte_fallback_ids[b] = (int32_t)id;
        }
    } else if (tok->type == TOKENIZER_SENTENCEPIECE_BPE) {
        for (int b = 0; b < 256; b++) {
            char buf[8];
            int  blen = snprintf(buf, sizeof(buf), "<0x%02X>", b);
            uint32_t id;
            if (str_u32_map_get(&tok->vocab, buf, (uint32_t)blen, &id))
                tok->byte_fallback_ids[b] = (int32_t)id;
        }
    }

    tok->id_to_token_cap = tok->vocab.count ? max_id + 1 : 0;
    if (tok->id_to_token_cap) {
        tok->id_to_token =
            (const char **)calloc(tok->id_to_token_cap, sizeof(*tok->id_to_token));
        if (!tok->id_to_token) {
            tokenizer_free(tok);
            return NULL;
        }
        /* Fill from the vocab map rather than re-walking the JSON: same
         * borrowed NUL-terminated key pointers, ids already validated. */
        str_u32_map_foreach(&tok->vocab, e) tok->id_to_token[e->val] = e->ptr;
    }

    if (merge_count) {
        yyjson_val *entry;
        yyjson_arr_foreach(merges_val, idx, max, entry) {
            const char *l, *r;
            uint32_t    llen, rlen;
            if (!parse_merge_pair(entry, &l, &llen, &r, &rlen)) continue;
            if (!merge_map_put(&tok->merges, l, llen, r, rlen, (uint32_t)idx)) {
                tokenizer_free(tok);
                return NULL;
            }
        }
    }

    return tok;
}

/* --- BPE merge (heap + linked list) ---------------------------------------- */

void encode_scratch_init(encode_scratch *s) { memset(s, 0, sizeof(*s)); }

bool encode_scratch_reserve(encode_scratch *s, size_t input_len) {
    /* heap_cap = 3 * len is uint32_t arithmetic: refuse any len that would
     * wrap it. UINT32_MAX / 3 < INT32_MAX, so this also keeps node indices
     * representable as int32_t. */
    if (input_len > UINT32_MAX / 3) return false;
    uint32_t len = input_len ? (uint32_t)input_len : 1;
    if (s->nodes_cap < len) {
        bpe_node *nodes = realloc(s->nodes, (size_t)len * sizeof(*s->nodes));
        if (!nodes) return false;
        s->nodes     = nodes;
        s->nodes_cap = len;
    }
    if (s->ids_cap < len) {
        int32_t *ids = realloc(s->ids, (size_t)len * sizeof(*s->ids));
        if (!ids) return false;
        s->ids     = ids;
        s->ids_cap = len;
    }
    if (s->heap_cap / 3 < len) {
        bpe_cand *heap = realloc(s->heap, 3 * (size_t)len * sizeof(*s->heap));
        if (!heap) return false;
        s->heap     = heap;
        s->heap_cap = 3 * len;
    }
    if (s->pretoks_cap < len) {
        pretok_slice *pretoks = realloc(s->pretoks, (size_t)len * sizeof(*s->pretoks));
        if (!pretoks) return false;
        s->pretoks     = pretoks;
        s->pretoks_cap = len;
    }
    return true;
}

void encode_scratch_free(encode_scratch *s) {
    free(s->nodes);
    free(s->heap);
    free(s->ids);
    free(s->pretoks);
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
    /* Total pushes are bounded by 3*(n_nodes-1) < heap_cap = 3*len: n_nodes-1
     * seeds plus at most two re-pushes per merge, and merges never add nodes. */
    assert(s->heap_len < s->heap_cap);
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

/* Emit vocab IDs for one surviving symbol into ids[count...]: a direct vocab
 * hit, else the tokenizer-type-specific per-byte fallback. Returns the new
 * count. */
static int bpe_emit_symbol(const tokenizer_t *tok, const char *input, const bpe_node *n,
                           int32_t *ids, int count) {
    uint32_t id;
    if (str_u32_map_get(&tok->vocab, input + n->start, n->end - n->start, &id)) {
        ids[count++] = (int32_t)id;
        return count;
    }
    /* Per-byte fallback: ids were resolved at load per tokenizer type (BPE
     * byte-to-unicode tokens, SentencePiece <0xNN>). A byte without a vocab
     * entry is pathological (e.g. truncated vocab); emit <unk> when the vocab
     * has one rather than silently dropping the byte. */
    for (uint32_t b = n->start; b < n->end; b++) {
        int32_t fid = tok->byte_fallback_ids[(uint8_t)input[b]];
        if (fid >= 0)
            ids[count++] = fid;
        else if (tok->unk_id >= 0)
            ids[count++] = tok->unk_id;
    }
    return count;
}

int bpe_merge(const tokenizer_t *tok, encode_scratch *s, const char *input, size_t len,
              int32_t **out) {
    /* WordPiece is greedy longest-match, not pair merging; routing it through
     * here would mis-tokenize via the SentencePiece byte fallback. */
    if (tok->type == TOKENIZER_WORDPIECE) return -1;
    /* Node indices are int32_t and positions uint32_t: longer inputs are
     * unrepresentable (and a truncated cast below would loop forever). */
    if (len > (size_t)INT32_MAX) return -1;

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
    for (int32_t cur = n_nodes > 0 ? 0 : -1; cur >= 0; cur = s->nodes[cur].next)
        count = bpe_emit_symbol(tok, input, &s->nodes[cur], s->ids, count);
    *out = s->ids;
    return count;
}

/* --- GPT-2 pre-tokenization ------------------------------------------------- */

/* Splits text following the Qwen / Llama-3 / GPT-2 pre-tokenizer regex as a
 * hand-rolled state machine. Each iteration picks the FIRST matching pattern
 * from the alternation, in declared order.
 *
 * Reference (from tokenizer.json pre_tokenizer.pretokenizers[0].pattern):
 *
 *     (?i:'s|'t|'re|'ve|'m|'ll|'d)
 *   | [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
 *   | \p{N}
 *   |  ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
 *   | \s*[\r\n]+
 *   | \s+(?!\S)
 *   | \s+
 */

/* Pattern 1: contraction `(?i:'s|'t|'re|'ve|'m|'ll|'d)`. Returns end
 * position of the match, or i if no contraction starts at i. */
static uint32_t match_contraction(const uint8_t *text, uint32_t len, uint32_t i) {
    if (text[i] != '\'' || i + 1 >= len) return i;
    uint8_t next = (uint8_t)tolower(text[i + 1]);
    if (next == 's' || next == 't' || next == 'm' || next == 'd') return i + 2;
    if (i + 2 < len) {
        uint8_t next2 = (uint8_t)tolower(text[i + 2]);
        if ((next == 'r' && next2 == 'e') || (next == 'v' && next2 == 'e') ||
            (next == 'l' && next2 == 'l'))
            return i + 3;
    }
    return i;
}

/* Consume a `[\p{L}\p{M}]+` run starting at i; returns its end (== i when
 * text[i] is not a letter/mark). */
static uint32_t scan_letter_mark_run(const uint8_t *text, uint32_t len, uint32_t i) {
    uint32_t end = i;
    while (end < len) {
        uc_cp_info c = uc_decode_codepoint(text, len, end);
        if (!uc_is_letter_or_mark(c.cp)) break;
        end += c.len;
    }
    return end;
}

/* Pattern 2: `[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+`. The optional codepoint may
 * be whitespace or punct - anything but \r, \n, letter, number. Returns end
 * position of the match, or i if no letters at the right place. */
static uint32_t match_letters_run(const uint8_t *text, uint32_t len, uint32_t i) {
    uc_cp_info first = uc_decode_codepoint(text, len, i);

    /* Try with the optional non-LNN codepoint consumed. */
    if (!uc_is_letter(first.cp) && !uc_is_number(first.cp) && first.cp != '\r' &&
        first.cp != '\n') {
        uint32_t after_opt = i + first.len;
        uint32_t end       = scan_letter_mark_run(text, len, after_opt);
        if (end > after_opt) return end;
    }

    /* Try with the optional codepoint empty: i itself must start the run. */
    if (uc_is_letter_or_mark(first.cp)) return scan_letter_mark_run(text, len, i);

    return i;
}

/* Pattern 4: ` ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*`. The optional space MUST be
 * exactly the byte 0x20 (regex literal), not any other whitespace. Returns
 * end position of the match, or i if no punct/symbol run at the right
 * place. */
static uint32_t match_space_punct(const uint8_t *text, uint32_t len, uint32_t i) {
    uint32_t p = i;
    if (text[p] == ' ') p++;
    if (p >= len) return i;

    uint32_t end = p;
    while (end < len) {
        uc_cp_info c = uc_decode_codepoint(text, len, end);
        if (uc_is_whitespace_cp(c.cp) || uc_is_letter_or_mark(c.cp) || uc_is_number(c.cp))
            break;
        end += c.len;
    }
    if (end == p) return i;
    while (end < len && (text[end] == '\r' || text[end] == '\n')) end++;
    return end;
}

int gpt2_pretokenize(encode_scratch *s, const char *input, size_t len) {
    if (len > INT32_MAX) return -1;
    const uint8_t *text  = (const uint8_t *)input;
    uint32_t       tlen  = (uint32_t)len;
    uint32_t       i     = 0;
    int            count = 0;
    while (i < tlen) {
        uint32_t end;

        /* Pattern 1: contraction. */
        end = match_contraction(text, tlen, i);

        /* Pattern 2: optional non-LNN char + letter/mark run. */
        if (end == i) end = match_letters_run(text, tlen, i);

        /* Pattern 3: exactly ONE \p{N} codepoint (no +). */
        if (end == i) {
            uc_cp_info c = uc_decode_codepoint(text, tlen, i);
            if (uc_is_number(c.cp)) end = i + c.len;
        }

        /* Pattern 4: optional literal space + punct run + [\r\n]* tail. */
        if (end == i) end = match_space_punct(text, tlen, i);

        /* Fallback: single byte, so the scan always advances. */
        if (end == i) end = i + 1;

        s->pretoks[count].off = i;
        s->pretoks[count].len = end - i;
        count++;
        i = end;
    }
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
