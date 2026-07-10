#include "model/tok_map.h"

#include <stdlib.h>
#include <string.h>

#define FNV_INIT 0xcbf29ce484222325ULL
#define FNV_PRIME 0x100000001b3ULL

static uint64_t fnv1a_update(uint64_t h, const char *data, uint32_t len) {
    for (uint32_t i = 0; i < len; i++) {
        h ^= (uint8_t)data[i];
        h *= FNV_PRIME;
    }
    return h;
}

static uint64_t fnv1a(const char *data, uint32_t len) {
    return fnv1a_update(FNV_INIT, data, len);
}

static uint32_t next_pow2(uint32_t v) {
    if (v == 0)
        return 1;
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return v + 1;
}

/* --- str_u32_map ---------------------------------------------------------- */

void str_u32_map_init(str_u32_map *m, uint32_t initial_cap) {
    m->cap = next_pow2(initial_cap < 4 ? 4 : initial_cap);
    m->count = 0;
    m->entries = calloc(m->cap, sizeof(str_u32_entry));
}

void str_u32_map_free(str_u32_map *m) {
    free(m->entries);
    m->entries = NULL;
    m->cap = m->count = 0;
}

static void str_u32_map_grow(str_u32_map *m) {
    uint32_t new_cap = m->cap * 2;
    str_u32_entry *new_entries = calloc(new_cap, sizeof(str_u32_entry));
    uint32_t mask = new_cap - 1;
    for (uint32_t i = 0; i < m->cap; i++) {
        str_u32_entry *e = &m->entries[i];
        if (!e->ptr)
            continue;
        uint32_t idx = (uint32_t)(e->hash & mask);
        while (new_entries[idx].ptr)
            idx = (idx + 1) & mask;
        new_entries[idx] = *e;
    }
    free(m->entries);
    m->entries = new_entries;
    m->cap = new_cap;
}

void str_u32_map_put(str_u32_map *m, const char *key, uint32_t key_len, uint32_t val) {
    if (m->count * 4 >= m->cap * 3)
        str_u32_map_grow(m);
    uint64_t h = fnv1a(key, key_len);
    uint32_t mask = m->cap - 1;
    uint32_t idx = (uint32_t)(h & mask);
    for (;;) {
        str_u32_entry *e = &m->entries[idx];
        if (!e->ptr) {
            e->ptr = key;
            e->len = key_len;
            e->hash = h;
            e->val = val;
            m->count++;
            return;
        }
        if (e->hash == h && e->len == key_len && memcmp(e->ptr, key, key_len) == 0) {
            e->val = val;
            return;
        }
        idx = (idx + 1) & mask;
    }
}

bool str_u32_map_get(const str_u32_map *m, const char *key, uint32_t key_len,
                     uint32_t *out_val) {
    uint64_t h = fnv1a(key, key_len);
    uint32_t mask = m->cap - 1;
    uint32_t idx = (uint32_t)(h & mask);
    for (;;) {
        const str_u32_entry *e = &m->entries[idx];
        if (!e->ptr)
            return false;
        if (e->hash == h && e->len == key_len && memcmp(e->ptr, key, key_len) == 0) {
            *out_val = e->val;
            return true;
        }
        idx = (idx + 1) & mask;
    }
}

/* --- merge_map ------------------------------------------------------------ */

static uint64_t merge_hash(const char *l, uint32_t llen, const char *r, uint32_t rlen) {
    uint64_t h = fnv1a_update(FNV_INIT, l, llen);
    h ^= 0xFF;
    h *= FNV_PRIME;
    return fnv1a_update(h, r, rlen);
}

static bool merge_eq(const merge_entry *e, const char *l, uint32_t llen,
                     const char *r, uint32_t rlen) {
    return e->llen == llen && e->rlen == rlen &&
           memcmp(e->l, l, llen) == 0 && memcmp(e->r, r, rlen) == 0;
}

void merge_map_init(merge_map *m, uint32_t initial_cap) {
    m->cap = next_pow2(initial_cap < 4 ? 4 : initial_cap);
    m->count = 0;
    m->entries = calloc(m->cap, sizeof(merge_entry));
}

void merge_map_free(merge_map *m) {
    free(m->entries);
    m->entries = NULL;
    m->cap = m->count = 0;
}

static void merge_map_grow(merge_map *m) {
    uint32_t new_cap = m->cap * 2;
    merge_entry *new_entries = calloc(new_cap, sizeof(merge_entry));
    uint32_t mask = new_cap - 1;
    for (uint32_t i = 0; i < m->cap; i++) {
        merge_entry *e = &m->entries[i];
        if (!e->l)
            continue;
        uint32_t idx = (uint32_t)(e->hash & mask);
        while (new_entries[idx].l)
            idx = (idx + 1) & mask;
        new_entries[idx] = *e;
    }
    free(m->entries);
    m->entries = new_entries;
    m->cap = new_cap;
}

void merge_map_put(merge_map *m, const char *l, uint32_t llen, const char *r,
                   uint32_t rlen, uint32_t rank) {
    if (m->count * 4 >= m->cap * 3)
        merge_map_grow(m);
    uint64_t h = merge_hash(l, llen, r, rlen);
    uint32_t mask = m->cap - 1;
    uint32_t idx = (uint32_t)(h & mask);
    for (;;) {
        merge_entry *e = &m->entries[idx];
        if (!e->l) {
            e->l = l;
            e->r = r;
            e->llen = llen;
            e->rlen = rlen;
            e->hash = h;
            e->rank = rank;
            m->count++;
            return;
        }
        if (e->hash == h && merge_eq(e, l, llen, r, rlen)) {
            e->rank = rank;
            return;
        }
        idx = (idx + 1) & mask;
    }
}

bool merge_map_get(const merge_map *m, const char *l, uint32_t llen,
                   const char *r, uint32_t rlen, uint32_t *out_rank) {
    uint64_t h = merge_hash(l, llen, r, rlen);
    uint32_t mask = m->cap - 1;
    uint32_t idx = (uint32_t)(h & mask);
    for (;;) {
        const merge_entry *e = &m->entries[idx];
        if (!e->l)
            return false;
        if (e->hash == h && merge_eq(e, l, llen, r, rlen)) {
            *out_rank = e->rank;
            return true;
        }
        idx = (idx + 1) & mask;
    }
}
