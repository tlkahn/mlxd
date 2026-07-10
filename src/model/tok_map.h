#ifndef MLXD_TOK_MAP_H
#define MLXD_TOK_MAP_H

#include <stdbool.h>
#include <stdint.h>

typedef struct {
    const char *ptr;
    uint32_t    len;
    uint64_t    hash;
    uint32_t    val;
} str_u32_entry;

typedef struct {
    str_u32_entry *entries;
    uint32_t       cap;
    uint32_t       count;
} str_u32_map;

typedef struct {
    const char *l;
    const char *r;
    uint32_t    llen;
    uint32_t    rlen;
    uint64_t    hash;
    uint32_t    rank;
} merge_entry;

typedef struct {
    merge_entry *entries;
    uint32_t     cap;
    uint32_t     count;
} merge_map;

void str_u32_map_init(str_u32_map *m, uint32_t initial_cap);
void str_u32_map_free(str_u32_map *m);
void str_u32_map_put(str_u32_map *m, const char *key, uint32_t key_len, uint32_t val);
bool str_u32_map_get(const str_u32_map *m, const char *key, uint32_t key_len,
                     uint32_t *out_val);

void merge_map_init(merge_map *m, uint32_t initial_cap);
void merge_map_free(merge_map *m);
void merge_map_put(merge_map *m, const char *l, uint32_t llen, const char *r,
                   uint32_t rlen, uint32_t rank);
bool merge_map_get(const merge_map *m, const char *l, uint32_t llen,
                   const char *r, uint32_t rlen, uint32_t *out_rank);

#endif
