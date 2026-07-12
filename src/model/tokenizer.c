#include "model/tokenizer.h"

#include "model/tok_bpe.h"
#include "model/tok_map.h"
#include "model/tok_unicode.h"

#include <yyjson/yyjson.h>

#include <assert.h>
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
    /* Membership set (val unused) of added_tokens[].content with
     * "special": true; keys borrowed from doc. Decode drops members. */
    str_u32_map        special_tokens;
    const char       **id_to_token;
    uint32_t           id_to_token_cap;
    int                vocab_size;
    int32_t            bos_id;
    int32_t            eos_id;
    int32_t            unk_id; /* vocab id of "<unk>", -1 if absent */
    /* WordPiece continuation prefix (HF model.continuing_subword_prefix),
     * default "##"; borrowed from doc when overridden. */
    const char        *wp_prefix;
    uint32_t           wp_prefix_len;
    /* Per-byte fallback token id, resolved once at load (-1 = no vocab entry):
     * BPE maps bytes through the GPT-2 byte-to-unicode table, SentencePiece
     * through <0xNN> tokens, WordPiece has no byte fallback. */
    int32_t            byte_fallback_ids[256];
    /* GPT-2 byte-to-unicode table, built once at load for every tokenizer
     * type: decode_byte_level is callable regardless of type. */
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
    tok->doc           = doc;
    tok->bos_id        = -1;
    tok->eos_id        = -1;
    tok->unk_id        = -1;
    tok->wp_prefix     = "##";
    tok->wp_prefix_len = 2;
    yyjson_val *cs_prefix = yyjson_obj_get(model, "continuing_subword_prefix");
    if (yyjson_is_str(cs_prefix)) {
        tok->wp_prefix     = yyjson_get_str(cs_prefix);
        tok->wp_prefix_len = (uint32_t)yyjson_get_len(cs_prefix);
    }

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
    bool vocab_ok    = str_u32_map_init(&tok->vocab, (uint32_t)vocab_hint);
    bool merges_ok   = merge_map_init(&tok->merges, (uint32_t)merge_hint);
    bool specials_ok = str_u32_map_init(&tok->special_tokens, 16);
    if (!vocab_ok || !merges_ok || !specials_ok) {
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

    yyjson_val *added = yyjson_obj_get(root, "added_tokens");
    if (yyjson_is_arr(added)) {
        yyjson_val *entry;
        yyjson_arr_foreach(added, idx, max, entry) {
            yyjson_val *content = yyjson_obj_get(entry, "content");
            if (!yyjson_is_str(content)) continue;
            if (!yyjson_is_true(yyjson_obj_get(entry, "special"))) continue;
            if (!str_u32_map_put(&tok->special_tokens, yyjson_get_str(content),
                                 (uint32_t)yyjson_get_len(content), 1)) {
                tokenizer_free(tok);
                return NULL;
            }
        }
    }

    /* Stage-E-minimal special-id resolution: WordPiece names its specials
     * [CLS]/[SEP]/[UNK]. Stage G generalizes this across tokenizer types. */
    if (tok->type == TOKENIZER_WORDPIECE) {
        uint32_t id;
        if (str_u32_map_get(&tok->vocab, "[CLS]", 5, &id)) tok->bos_id = (int32_t)id;
        if (str_u32_map_get(&tok->vocab, "[SEP]", 5, &id)) tok->eos_id = (int32_t)id;
        if (str_u32_map_get(&tok->vocab, "[UNK]", 5, &id)) tok->unk_id = (int32_t)id;
    }

    /* Resolve the per-byte fallback ids once, now that the vocab map is
     * built; bpe_emit_symbol then indexes instead of re-hashing per byte. */
    uc_build_bytes_to_unicode(&tok->bytes_unicode);
    for (int b = 0; b < 256; b++) tok->byte_fallback_ids[b] = -1;
    if (tok->type == TOKENIZER_BPE) {
        for (int b = 0; b < 256; b++) {
            char buf[4];
            /* The byte table's max mapped codepoint is 323, so this always
             * encodes to 1-2 bytes and never fails. */
            uint32_t blen = uc_encode_codepoint(tok->bytes_unicode.byte_to_cp[b], buf);
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

/* One bit per scratch buffer, so each encoder reserves only what its mode
 * touches (WordPiece never merges; SentencePiece never pre-tokenizes). */
enum {
    SCRATCH_NODES   = 1u << 0,
    SCRATCH_HEAP    = 1u << 1,
    SCRATCH_IDS     = 1u << 2,
    SCRATCH_PRETOKS = 1u << 3,
    SCRATCH_TEXT    = 1u << 4,
    SCRATCH_OUT     = 1u << 5,
    SCRATCH_CAND    = 1u << 6,
    SCRATCH_ALL     = (1u << 7) - 1,
};

static bool scratch_reserve_mask(encode_scratch *s, size_t input_len, unsigned mask) {
    /* heap_cap = 3 * len is uint32_t arithmetic: refuse any len that would
     * wrap it. UINT32_MAX / 3 < INT32_MAX, so this also keeps node indices
     * representable as int32_t. */
    if (input_len > UINT32_MAX / 3) return false;
    uint32_t len = input_len ? (uint32_t)input_len : 1;
    if ((mask & SCRATCH_NODES) && s->nodes_cap < len) {
        bpe_node *nodes = realloc(s->nodes, (size_t)len * sizeof(*s->nodes));
        if (!nodes) return false;
        s->nodes     = nodes;
        s->nodes_cap = len;
    }
    if ((mask & SCRATCH_IDS) && s->ids_cap < len) {
        int32_t *ids = realloc(s->ids, (size_t)len * sizeof(*s->ids));
        if (!ids) return false;
        s->ids     = ids;
        s->ids_cap = len;
    }
    if ((mask & SCRATCH_HEAP) && s->heap_cap / 3 < len) {
        bpe_cand *heap = realloc(s->heap, 3 * (size_t)len * sizeof(*s->heap));
        if (!heap) return false;
        s->heap     = heap;
        s->heap_cap = 3 * len;
    }
    if ((mask & SCRATCH_PRETOKS) && s->pretoks_cap < len) {
        pretok_slice *pretoks = realloc(s->pretoks, (size_t)len * sizeof(*s->pretoks));
        if (!pretoks) return false;
        s->pretoks     = pretoks;
        s->pretoks_cap = len;
    }
    if ((mask & SCRATCH_TEXT) && s->text_cap < len) {
        char *text = realloc(s->text, len);
        if (!text) return false;
        s->text     = text;
        s->text_cap = len;
    }
    if ((mask & SCRATCH_OUT) && s->out_cap < len) {
        int32_t *out = realloc(s->out, (size_t)len * sizeof(*s->out));
        if (!out) return false;
        s->out     = out;
        s->out_cap = len;
    }
    if ((mask & SCRATCH_CAND) && s->cand_cap < len + 2) {
        char *cand = realloc(s->cand, (size_t)len + 2);
        if (!cand) return false;
        s->cand     = cand;
        s->cand_cap = len + 2;
    }
    return true;
}

bool encode_scratch_reserve(encode_scratch *s, size_t input_len) {
    return scratch_reserve_mask(s, input_len, SCRATCH_ALL);
}

void encode_scratch_free(encode_scratch *s) {
    free(s->nodes);
    free(s->heap);
    free(s->ids);
    free(s->pretoks);
    free(s->text);
    free(s->out);
    free(s->cand);
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
    /* ASCII case-fold; tolower() is locale-dependent per C11. */
    uint8_t next = (uint8_t)(text[i + 1] | 0x20);
    if (next == 's' || next == 't' || next == 'm' || next == 'd') return i + 2;
    if (i + 2 < len) {
        uint8_t next2 = (uint8_t)(text[i + 2] | 0x20);
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
 * be whitespace or punct - anything but \r, \n, letter, number. `first` is
 * the already-decoded codepoint at i; the caller guarantees it is not
 * \p{N} (a number cp dispatches to pattern 3 before this runs). Returns
 * end position of the match, or i if no letters at the right place. */
static uint32_t match_letters_run(const uint8_t *text, uint32_t len, uint32_t i,
                                  uc_cp_info first) {
    bool lm = uc_is_letter_or_mark(first.cp);

    /* A letter/mark at i starts the run itself; first is already
     * classified, so the scan resumes after it. This branch is
     * load-bearing, not an optimization: for a BARE mark (no letter/mark
     * following), the optional-char path below fails - nothing is left to
     * satisfy [\p{L}\p{M}]+ - and the mark would fall through to the final
     * fallback in gpt2_pretokenize, violating its unreachable contract. */
    if (lm) return scan_letter_mark_run(text, len, i + first.len);

    /* Otherwise try with the optional non-LNN codepoint consumed (first is
     * never \p{N} here, per the caller's dispatch). */
    if (first.cp != '\r' && first.cp != '\n') {
        uint32_t after_opt = i + first.len;
        uint32_t end       = scan_letter_mark_run(text, len, after_opt);
        if (end > after_opt) return end;
    }

    return i;
}

/* Pattern 4: ` ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*`. The optional space MUST be
 * exactly the byte 0x20 (regex literal), not any other whitespace. Returns
 * end position of the match, or i if no punct/symbol run at the right
 * place. */
static uint32_t match_space_punct(const uint8_t *text, uint32_t len, uint32_t i,
                                  uc_cp_info first) {
    uint32_t p = i;
    if (text[p] == ' ') p++;

    uint32_t end = p;
    while (end < len) {
        /* The `end == i` reuse of first only serves the no-space path: when
         * the optional space was consumed, first holds the SPACE's cp info,
         * but end starts at i+1 and only grows, so end == i is unreachable
         * and the stale first is never read. */
        uc_cp_info c = end == i ? first : uc_decode_codepoint(text, len, end);
        if (uc_is_whitespace_cp(c.cp) || uc_is_letter_or_mark(c.cp) || uc_is_number(c.cp))
            break;
        end += c.len;
    }
    if (end == p) return i;
    while (end < len && (text[end] == '\r' || text[end] == '\n')) end++;
    return end;
}

/* Patterns 5+6+7: `\s*[\r\n]+`, `\s+(?!\S)`, fallback `\s+`. One greedy
 * codepoint-level whitespace scan resolves the whole group. Pattern 5's
 * `\s*` may itself consume newlines and later whitespace, so its
 * backtracked match ends one past the LAST \r/\n of the run. That single
 * fact IS full regex backtracking: greedy `\s*` gives back characters only
 * until `[\r\n]+` can match, and since everything before the run's last
 * \r/\n is itself \s, backtracking always lands `[\r\n]+` on that last
 * newline - even when the `\s*` prefix contains earlier newlines (e.g.
 * `\n\t\n\t` matches through the second \n). A run without newlines
 * resolves the pattern-6 lookahead instead: at end of input the
 * full run stands, before \S a multi-cp run backtracks one CODEPOINT
 * (whitespace can be multi-byte, e.g. NBSP) so the trailing space gets
 * handed to pattern 2/4 on the next iteration, while a single-cp run
 * cannot satisfy the lookahead and falls through to pattern 7, which
 * matches the same single codepoint. Returns i if there is no whitespace
 * at i. */
static uint32_t match_ws_run(const uint8_t *text, uint32_t len, uint32_t i,
                             uc_cp_info first) {
    uint32_t end     = i;
    uint32_t last_cp = i; /* start offset of the run's last whitespace cp */
    uint32_t nl_end  = 0; /* one past the run's last \r/\n; 0 = none */
    while (end < len) {
        uc_cp_info c = end == i ? first : uc_decode_codepoint(text, len, end);
        if (!uc_is_whitespace_cp(c.cp)) break;
        if (c.cp == '\r' || c.cp == '\n') nl_end = end + c.len;
        last_cp = end;
        end += c.len;
    }
    if (end == i) return i;
    if (nl_end != 0) return nl_end; /* pattern 5: `\s*[\r\n]+` */
    if (end == len) return end;     /* pattern 6: end of input satisfies (?!\S) */
    /* text[end] is \S: backtrack one cp so the lookahead sees whitespace;
     * a single-cp run is pattern 7's `\s+` match instead. */
    if (last_cp > i) return last_cp;
    return end;
}

int gpt2_pretokenize(encode_scratch *s, const char *input, size_t len) {
    if (len > INT32_MAX) return -1;
    const uint8_t *text  = (const uint8_t *)input;
    uint32_t       tlen  = (uint32_t)len;
    uint32_t       i     = 0;
    int            count = 0;
    while (i < tlen) {
        /* Decode the codepoint at i once; every matcher below reuses it. */
        uc_cp_info first = uc_decode_codepoint(text, tlen, i);
        uint32_t   end;

        /* Pattern 1: contraction. */
        end = match_contraction(text, tlen, i);

        /* Patterns 2+3. \p{N} is disjoint from \p{L}\p{M} and excluded from
         * pattern 2's optional class, so a number cp can never start pattern
         * 2; testing pattern 3 (exactly ONE \p{N} codepoint, no +) first is
         * order-equivalent and avoids a second uc_is_number lookup inside
         * match_letters_run. */
        if (end == i) {
            if (uc_is_number(first.cp)) end = i + first.len;
            else end = match_letters_run(text, tlen, i, first);
        }

        /* Pattern 4: optional literal space + punct run + [\r\n]* tail. */
        if (end == i) end = match_space_punct(text, tlen, i, first);

        /* Patterns 5+6+7: `\s*[\r\n]+`, then `\s+(?!\S)` with `\s+` fallback. */
        if (end == i) end = match_ws_run(text, tlen, i, first);

        /* Fallback so the scan always advances. Unreachable today (patterns
         * 2-7 cover every codepoint class, and invalid UTF-8 decodes as
         * U+FFFD with len 1), but if it ever fires it must not split a
         * multi-byte codepoint. */
        if (end == i) end = i + first.len;

        /* Reserve contract: cap >= len, so count can never reach cap. The
         * guard survives NDEBUG builds where an assert would compile away. */
        if ((uint32_t)count >= s->pretoks_cap) return -1;
        s->pretoks[count].off = i;
        s->pretoks[count].len = end - i;
        count++;
        i = end;
    }
    return count;
}

/* --- Per-mode encoders ------------------------------------------------------- */

int encode_byte_level(const tokenizer_t *tok, encode_scratch *s, const char *text, size_t len,
                      int32_t **out) {
    if (len > (size_t)INT32_MAX) return -1;
    /* 2*len: each input byte expands to a 1-2 byte codepoint in s->text, and
     * bpe_merge on that expansion needs nodes/ids sized to its byte length.
     * Byte-level uses every buffer except cand (no WordPiece key building). */
    if (!scratch_reserve_mask(s, 2 * len,
                              SCRATCH_NODES | SCRATCH_HEAP | SCRATCH_IDS | SCRATCH_PRETOKS |
                                  SCRATCH_TEXT | SCRATCH_OUT))
        return -1;

    int n_words = gpt2_pretokenize(s, text, len);
    if (n_words < 0) return -1;

    const uc_bytes_unicode_t *bu = &tok->bytes_unicode;

    int total = 0;
    for (int w = 0; w < n_words; w++) {
        pretok_slice word = s->pretoks[w];
        uint32_t     tlen = 0;
        for (uint32_t b = 0; b < word.len; b++) {
            /* Byte-table codepoints top out at 323: always 1-2 bytes. */
            tlen += uc_encode_codepoint(bu->byte_to_cp[(uint8_t)text[word.off + b]],
                                        s->text + tlen);
        }
        int32_t *ids;
        int      n = bpe_merge(tok, s, s->text, tlen, &ids);
        if (n < 0) return -1;
        /* Accumulate: bpe_merge reuses s->ids from offset 0 on every call.
         * Total ids across words <= total expanded bytes <= 2*len = out_cap. */
        memcpy(s->out + total, ids, (size_t)n * sizeof(*ids));
        total += n;
    }
    *out = s->out;
    return total;
}

int encode_sentencepiece(const tokenizer_t *tok, encode_scratch *s, const char *text,
                         size_t len, int32_t **out) {
    if (len > (size_t)INT32_MAX) return -1;
    /* 3*len: each ' ' becomes the 3-byte U+2581 in s->text, and bpe_merge on
     * that expansion needs nodes/ids sized to its byte length. No
     * pre-tokenization and ids come back via bpe_merge's s->ids, so pretoks,
     * out, and cand stay untouched. */
    if (!scratch_reserve_mask(s, 3 * len,
                              SCRATCH_NODES | SCRATCH_HEAP | SCRATCH_IDS | SCRATCH_TEXT))
        return -1;

    uint32_t tlen = 0;
    for (size_t i = 0; i < len; i++) {
        if (text[i] == ' ') {
            memcpy(s->text + tlen, "\xe2\x96\x81", 3);
            tlen += 3;
        } else {
            s->text[tlen++] = text[i];
        }
    }
    return bpe_merge(tok, s, s->text, tlen, out);
}

/* BERT punctuation: the four ASCII symbol runs around the alphanumerics. */
static bool wp_is_punct(uint8_t c) {
    return (c >= 0x21 && c <= 0x2F) || (c >= 0x3A && c <= 0x40) ||
           (c >= 0x5B && c <= 0x60) || (c >= 0x7B && c <= 0x7E);
}

/* Greedy longest-match one lowercased word into s->out at *total: probe
 * word[pos..end] (prefixed tok->wp_prefix via s->cand when pos > 0), shrink
 * end by one UTF-8 char on a miss; a word with no match at any pos is bad and
 * becomes a single unk, discarding pieces already emitted for it (BERT
 * is_bad). */
static void wp_emit_word(const tokenizer_t *tok, encode_scratch *s, const char *word,
                         uint32_t wlen, int32_t unk, int *total) {
    int      start = *total;
    uint32_t pos   = 0;
    while (pos < wlen) {
        uint32_t end   = wlen;
        bool     found = false;
        /* The probe key depends only on pos: build prefix + word[pos..wlen]
         * once, shrinking end just shortens klen. */
        const char *key;
        uint32_t    plen;
        if (pos > 0) {
            memcpy(s->cand, tok->wp_prefix, tok->wp_prefix_len);
            memcpy(s->cand + tok->wp_prefix_len, word + pos, wlen - pos);
            key  = s->cand;
            plen = tok->wp_prefix_len;
        } else {
            key  = word + pos;
            plen = 0;
        }
        while (end > pos) {
            uint32_t klen = end - pos + plen;
            uint32_t id;
            if (str_u32_map_get(&tok->vocab, key, klen, &id)) {
                s->out[(*total)++] = (int32_t)id;
                pos   = end;
                found = true;
                break;
            }
            /* Shrink by one UTF-8 character: skip continuation bytes. */
            end--;
            while (end > pos && ((uint8_t)word[end] & 0xC0) == 0x80) end--;
        }
        if (!found) {
            *total             = start;
            s->out[(*total)++] = unk;
            return;
        }
    }
}

int encode_wordpiece(const tokenizer_t *tok, encode_scratch *s, const char *text, size_t len,
                     int32_t **out) {
    if (len > (size_t)INT32_MAX) return -1;
    /* len + 2 covers out (at most len body ids plus the bos/eos wrap);
     * + wp_prefix_len covers cand's worst case, prefix + whole word.
     * Greedy longest-match, so nodes/heap/ids/pretoks stay untouched. */
    if (!scratch_reserve_mask(s, len + 2 + tok->wp_prefix_len,
                              SCRATCH_TEXT | SCRATCH_OUT | SCRATCH_CAND))
        return -1;

    int total = 0;
    /* Stage F seam: F4 moves this wrap to the public tokenizer_encode. */
    if (tok->bos_id >= 0) s->out[total++] = tok->bos_id;

    for (size_t i = 0; i < len; i++) {
        char c     = text[i];
        s->text[i] = (char)((c >= 'A' && c <= 'Z') ? c + 32 : c);
    }

    int32_t unk = tok->unk_id >= 0 ? tok->unk_id : 0;

    /* Split: ASCII whitespace drops, each punct byte is its own word. */
    size_t wstart = 0;
    for (size_t i = 0; i < len; i++) {
        char c = s->text[i];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            if (i > wstart)
                wp_emit_word(tok, s, s->text + wstart, (uint32_t)(i - wstart), unk, &total);
            wstart = i + 1;
        } else if (wp_is_punct((uint8_t)c)) {
            if (i > wstart)
                wp_emit_word(tok, s, s->text + wstart, (uint32_t)(i - wstart), unk, &total);
            wp_emit_word(tok, s, s->text + i, 1, unk, &total);
            wstart = i + 1;
        }
    }
    if (wstart < len) wp_emit_word(tok, s, s->text + wstart, (uint32_t)(len - wstart), unk, &total);

    if (tok->eos_id >= 0) s->out[total++] = tok->eos_id;
    *out = s->out;
    return total;
}

/* --- Per-mode decoders ------------------------------------------------------- */

/* Realloc-doubling byte accumulator for the decoders. buf is NUL-terminated
 * after every successful append; a NULL buf means nothing appended yet. */
typedef struct {
    char  *buf;
    size_t len;
    size_t cap;
} strbuf;

static bool sb_append(strbuf *sb, const char *bytes, size_t n) {
    /* len + n + 1 must not wrap size_t. */
    if (n >= SIZE_MAX - sb->len) return false;
    if (sb->len + n + 1 > sb->cap) {
        size_t cap = sb->cap ? sb->cap : 16;
        while (cap < sb->len + n + 1) {
            if (cap > SIZE_MAX / 2) return false;
            cap *= 2;
        }
        char *buf = realloc(sb->buf, cap);
        if (!buf) return false;
        sb->buf = buf;
        sb->cap = cap;
    }
    memcpy(sb->buf + sb->len, bytes, n);
    sb->len += n;
    sb->buf[sb->len] = '\0';
    return true;
}

/* Hand ownership of the accumulated string to the caller; an untouched
 * strbuf becomes a malloc'd empty string so decoders never return NULL for
 * an empty result. */
static char *sb_finish(strbuf *sb) {
    if (sb->buf) return sb->buf;
    char *s = malloc(1);
    if (s) s[0] = '\0';
    return s;
}

char *decode_byte_level(const tokenizer_t *tok, const int32_t *ids, int count) {
    const uc_bytes_unicode_t *bu  = &tok->bytes_unicode;
    strbuf                    out = {0};
    /* One pass per token, no concat buffer: vocab tokens are yyjson-validated
     * UTF-8, so a codepoint never spans a token boundary and scanning each
     * token alone equals scanning their concatenation. */
    for (int i = 0; i < count; i++) {
        const char *token = tokenizer_decode_token(tok, ids[i]);
        if (!token) continue;
        size_t slen = strlen(token);
        if (slen > UINT32_MAX) {
            free(out.buf);
            return NULL;
        }
        const uint8_t *text = (const uint8_t *)token;
        uint32_t       tlen = (uint32_t)slen;
        uint32_t       j    = 0;
        while (j < tlen) {
            /* Invalid UTF-8 decodes as U+FFFD with len 1: above the byte
             * table's range, so the raw-bytes branch passes the byte through
             * unchanged. */
            uc_cp_info c = uc_decode_codepoint(text, tlen, j);
            bool       ok;
            if (c.cp < UC_BYTES_UNICODE_REV_SIZE && bu->cp_to_byte[c.cp] != UINT16_MAX) {
                char b = (char)(uint8_t)bu->cp_to_byte[c.cp];
                ok     = sb_append(&out, &b, 1);
            } else {
                ok = sb_append(&out, (const char *)text + j, c.len);
            }
            if (!ok) {
                free(out.buf);
                return NULL;
            }
            j += c.len;
        }
    }
    return sb_finish(&out);
}

char *decode_wordpiece(const tokenizer_t *tok, const int32_t *ids, int count) {
    strbuf out = {0};
    for (int i = 0; i < count; i++) {
        const char *token = tokenizer_decode_token(tok, ids[i]);
        if (!token) continue;
        size_t tlen = strlen(token);
        /* Registered specials ([CLS], [SEP], ...) are dropped; ordinary
         * vocab tokens that merely start with '[' are kept. */
        uint32_t one;
        if (str_u32_map_get(&tok->special_tokens, token, (uint32_t)tlen, &one)) continue;
        bool     ok;
        uint32_t plen = tok->wp_prefix_len;
        if (tlen >= plen && memcmp(token, tok->wp_prefix, plen) == 0) {
            ok = sb_append(&out, token + plen, tlen - plen);
        } else {
            ok = (i == 0 || out.len == 0 || sb_append(&out, " ", 1)) &&
                 sb_append(&out, token, tlen);
        }
        if (!ok) {
            free(out.buf);
            return NULL;
        }
    }
    return sb_finish(&out);
}

/* Parse a <0xNN> byte-fallback token: exactly len 6, "<0x", two hex digits,
 * ">". Returns the byte value, or -1 when the token is not that shape. */
static int sp_byte_fallback(const char *token, size_t len) {
    if (len != 6 || memcmp(token, "<0x", 3) != 0 || token[5] != '>') return -1;
    int v = 0;
    for (int i = 3; i < 5; i++) {
        char c = token[i];
        int  d;
        if (c >= '0' && c <= '9') d = c - '0';
        else if (c >= 'A' && c <= 'F') d = c - 'A' + 10;
        else if (c >= 'a' && c <= 'f') d = c - 'a' + 10;
        else return -1;
        v = v * 16 + d;
    }
    return v;
}

char *decode_sentencepiece(const tokenizer_t *tok, const int32_t *ids, int count,
                           bool strip_leading_space) {
    strbuf raw = {0};
    for (int i = 0; i < count; i++) {
        const char *token = tokenizer_decode_token(tok, ids[i]);
        if (!token) continue;
        size_t tlen = strlen(token);
        int    b    = sp_byte_fallback(token, tlen);
        bool   ok;
        if (b >= 0) {
            char c = (char)(uint8_t)b;
            ok     = sb_append(&raw, &c, 1);
        } else {
            ok = sb_append(&raw, token, tlen);
        }
        if (!ok) {
            free(raw.buf);
            return NULL;
        }
    }

    if (raw.buf) {
        /* Rewrite every 3-byte U+2581 as ' ' in place; r + 2 < len still
         * admits a U+2581 occupying the final 3 bytes. */
        size_t w = 0, r = 0;
        while (r < raw.len) {
            if (r + 2 < raw.len && (uint8_t)raw.buf[r] == 0xE2 &&
                (uint8_t)raw.buf[r + 1] == 0x96 && (uint8_t)raw.buf[r + 2] == 0x81) {
                raw.buf[w++] = ' ';
                r += 3;
            } else {
                raw.buf[w++] = raw.buf[r++];
            }
        }
        raw.len = w;
        if (strip_leading_space && raw.len > 0 && raw.buf[0] == ' ') {
            memmove(raw.buf, raw.buf + 1, raw.len - 1);
            raw.len--;
        }
        raw.buf[raw.len] = '\0';
    }
    return sb_finish(&raw);
}

void tokenizer_free(tokenizer_t *tok) {
    if (!tok) return;
    str_u32_map_free(&tok->vocab);
    merge_map_free(&tok->merges);
    str_u32_map_free(&tok->special_tokens);
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
    if (!tok || count < 0 || (count > 0 && !ids)) return NULL;
    switch (tok->type) {
    case TOKENIZER_BPE:
        return decode_byte_level(tok, ids, count);
    case TOKENIZER_WORDPIECE:
        return decode_wordpiece(tok, ids, count);
    case TOKENIZER_SENTENCEPIECE_BPE:
        /* Leading-space stripping is a caller policy (chat template glue);
         * the public entry point always preserves it. */
        return decode_sentencepiece(tok, ids, count, false);
    }
    return NULL;
}

int tokenizer_vocab_size(const tokenizer_t *tok) { return tok ? tok->vocab_size : 0; }

int32_t tokenizer_bos_id(const tokenizer_t *tok) { return tok ? tok->bos_id : -1; }

int32_t tokenizer_eos_id(const tokenizer_t *tok) { return tok ? tok->eos_id : -1; }
