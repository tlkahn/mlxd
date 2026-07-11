/* Unit tests for tok_map's allocation-failure paths. Compiles a private,
 * fault-injectable copy of tok_map.c: public symbols are macro-renamed so they
 * don't collide with tok_map.o (linked into every test binary), and calloc is
 * routed through fi_calloc so grow/init failures are deterministic. */

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void *fi_calloc(size_t nmemb, size_t size);

#define calloc            fi_calloc
#define str_u32_map_init  fi_str_u32_map_init
#define str_u32_map_free  fi_str_u32_map_free
#define str_u32_map_put   fi_str_u32_map_put
#define str_u32_map_get   fi_str_u32_map_get
#define merge_map_init    fi_merge_map_init
#define merge_map_free    fi_merge_map_free
#define merge_map_put     fi_merge_map_put
#define merge_map_get     fi_merge_map_get
#include "model/tok_map.c"

/* Fault controls: fail_all fails every allocation; nmemb_limit (when nonzero)
 * fails any allocation of more than that many entries, so huge-capacity tests
 * never actually allocate. */
static bool   fi_fail_all;
static size_t fi_nmemb_limit;

static void *fi_calloc(size_t nmemb, size_t size) {
    if (fi_fail_all) return NULL;
    if (fi_nmemb_limit && nmemb > fi_nmemb_limit) return NULL;
    void *p = malloc(nmemb * size);
    if (p) memset(p, 0, nmemb * size);
    return p;
}

/* --- basic behavior (sanity for the fault-injected copy) ------------------- */

static void test_str_map_basic(void) {
    str_u32_map m;
    assert(str_u32_map_init(&m, 4));
    assert(m.cap == 4);
    assert(str_u32_map_put(&m, "a", 1, 10));
    uint32_t v = 0;
    assert(str_u32_map_get(&m, "a", 1, &v) && v == 10);
    assert(str_u32_map_put(&m, "a", 1, 11)); /* key update, count stays 1 */
    assert(str_u32_map_get(&m, "a", 1, &v) && v == 11);
    assert(m.count == 1);
    assert(!str_u32_map_get(&m, "b", 1, &v));
    str_u32_map_free(&m);
}

static void test_merge_map_basic(void) {
    merge_map m;
    assert(merge_map_init(&m, 4));
    assert(m.cap == 4);
    assert(merge_map_put(&m, "a", 1, "b", 1, 7));
    uint32_t r = 0;
    assert(merge_map_get(&m, "a", 1, "b", 1, &r) && r == 7);
    assert(!merge_map_get(&m, "b", 1, "a", 1, &r));
    merge_map_free(&m);
}

/* --- failed grow near a full table ------------------------------------------ */

/* A put that trips the grow threshold while allocation is failing must be
 * refused once the table is one-short-of-full: inserting into the last empty
 * slot would leave _get with no NULL sentinel, so a miss probes forever. */
static void test_str_map_put_refused_when_grow_fails_near_full(void) {
    str_u32_map m;
    assert(str_u32_map_init(&m, 4));
    assert(m.cap == 4);
    assert(str_u32_map_put(&m, "a", 1, 1));
    assert(str_u32_map_put(&m, "b", 1, 2));
    assert(str_u32_map_put(&m, "c", 1, 3));
    assert(m.count == 3 && m.cap == 4);

    fi_fail_all = true;
    assert(!str_u32_map_put(&m, "d", 1, 4));
    fi_fail_all = false;

    /* Existing keys survive the refused put; a miss terminates. */
    uint32_t v = 0;
    assert(str_u32_map_get(&m, "a", 1, &v) && v == 1);
    assert(str_u32_map_get(&m, "b", 1, &v) && v == 2);
    assert(str_u32_map_get(&m, "c", 1, &v) && v == 3);
    assert(!str_u32_map_get(&m, "d", 1, &v));

    /* With allocation working again the same put grows and succeeds. */
    assert(str_u32_map_put(&m, "d", 1, 4));
    assert(m.cap == 8);
    assert(str_u32_map_get(&m, "d", 1, &v) && v == 4);
    str_u32_map_free(&m);
}

static void test_merge_map_put_refused_when_grow_fails_near_full(void) {
    merge_map m;
    assert(merge_map_init(&m, 4));
    assert(m.cap == 4);
    assert(merge_map_put(&m, "a", 1, "b", 1, 0));
    assert(merge_map_put(&m, "b", 1, "c", 1, 1));
    assert(merge_map_put(&m, "c", 1, "d", 1, 2));
    assert(m.count == 3 && m.cap == 4);

    fi_fail_all = true;
    assert(!merge_map_put(&m, "d", 1, "e", 1, 3));
    fi_fail_all = false;

    uint32_t r = 0;
    assert(merge_map_get(&m, "a", 1, "b", 1, &r) && r == 0);
    assert(merge_map_get(&m, "b", 1, "c", 1, &r) && r == 1);
    assert(merge_map_get(&m, "c", 1, "d", 1, &r) && r == 2);
    assert(!merge_map_get(&m, "d", 1, "e", 1, &r));

    assert(merge_map_put(&m, "d", 1, "e", 1, 3));
    assert(m.cap == 8);
    assert(merge_map_get(&m, "d", 1, "e", 1, &r) && r == 3);
    merge_map_free(&m);
}

/* A value update of an existing key consumes no slot, so it must succeed even
 * when the grow attempt at the load threshold fails one-short-of-full; only a
 * fresh-key insert (which would fill the NULL sentinel) is refused. */
static void test_str_map_update_survives_failed_grow_near_full(void) {
    str_u32_map m;
    assert(str_u32_map_init(&m, 4));
    assert(m.cap == 4);
    assert(str_u32_map_put(&m, "a", 1, 1));
    assert(str_u32_map_put(&m, "b", 1, 2));
    assert(str_u32_map_put(&m, "c", 1, 3));
    assert(m.count == 3 && m.cap == 4);

    fi_fail_all = true;
    assert(str_u32_map_put(&m, "b", 1, 22)); /* update: no slot needed */
    uint32_t v = 0;
    assert(str_u32_map_get(&m, "b", 1, &v) && v == 22);
    assert(m.count == 3 && m.cap == 4);
    assert(!str_u32_map_put(&m, "d", 1, 4)); /* insert: still refused */
    assert(!str_u32_map_get(&m, "d", 1, &v)); /* miss still terminates */
    fi_fail_all = false;
    str_u32_map_free(&m);
}

static void test_merge_map_update_survives_failed_grow_near_full(void) {
    merge_map m;
    assert(merge_map_init(&m, 4));
    assert(m.cap == 4);
    assert(merge_map_put(&m, "a", 1, "b", 1, 0));
    assert(merge_map_put(&m, "b", 1, "c", 1, 1));
    assert(merge_map_put(&m, "c", 1, "d", 1, 2));
    assert(m.count == 3 && m.cap == 4);

    fi_fail_all = true;
    assert(merge_map_put(&m, "b", 1, "c", 1, 11)); /* rank update: no slot needed */
    uint32_t r = 0;
    assert(merge_map_get(&m, "b", 1, "c", 1, &r) && r == 11);
    assert(m.count == 3 && m.cap == 4);
    assert(!merge_map_put(&m, "d", 1, "e", 1, 3)); /* insert: still refused */
    assert(!merge_map_get(&m, "d", 1, "e", 1, &r)); /* miss still terminates */
    fi_fail_all = false;
    merge_map_free(&m);
}

/* --- foreach ----------------------------------------------------------------- */

/* str_u32_map_foreach must visit exactly the live entries, each once, with
 * the stored key/val - including after the table has grown and rehashed. */
static void test_str_map_foreach_visits_live_entries_once(void) {
    str_u32_map m;
    assert(str_u32_map_init(&m, 4));
    assert(m.cap == 4);

    char keys[6][8];
    for (int i = 0; i < 6; i++) {
        int len = snprintf(keys[i], sizeof(keys[i]), "k%d", i);
        assert(str_u32_map_put(&m, keys[i], (uint32_t)len, (uint32_t)(100 + i)));
    }
    assert(m.cap > 4); /* the inserts crossed at least one grow boundary */
    assert(m.count == 6);

    int seen[6] = {0};
    uint32_t visited = 0;
    str_u32_map_foreach(&m, e) {
        visited++;
        assert(e->len == 2 && e->ptr[0] == 'k');
        int i = e->ptr[1] - '0';
        assert(i >= 0 && i < 6);
        assert(memcmp(e->ptr, keys[i], 2) == 0);
        assert(e->val == (uint32_t)(100 + i));
        seen[i]++;
    }
    assert(visited == 6);
    for (int i = 0; i < 6; i++) assert(seen[i] == 1);

    str_u32_map_free(&m);
}

/* --- huge initial capacity --------------------------------------------------- */

/* next_pow2 wraps to 0 for v > 2^31; init must never hand back a "successful"
 * map with cap 0 (mask would be UINT32_MAX -> OOB probes). Either a clean
 * failure or a real nonzero capacity is acceptable. fi_nmemb_limit keeps the
 * test from ever allocating a real multi-GB table. */
static void test_str_map_init_huge_cap(void) {
    fi_nmemb_limit = 1u << 20;
    str_u32_map m;
    bool ok = str_u32_map_init(&m, UINT32_MAX);
    assert(!ok || m.cap != 0);
    str_u32_map_free(&m);
    ok = str_u32_map_init(&m, (1u << 31) + 1);
    assert(!ok || m.cap != 0);
    str_u32_map_free(&m);
    fi_nmemb_limit = 0;
}

static void test_merge_map_init_huge_cap(void) {
    fi_nmemb_limit = 1u << 20;
    merge_map m;
    bool ok = merge_map_init(&m, UINT32_MAX);
    assert(!ok || m.cap != 0);
    merge_map_free(&m);
    ok = merge_map_init(&m, (1u << 31) + 1);
    assert(!ok || m.cap != 0);
    merge_map_free(&m);
    fi_nmemb_limit = 0;
}

int main(void) {
    test_str_map_basic();
    test_merge_map_basic();
    test_str_map_put_refused_when_grow_fails_near_full();
    test_merge_map_put_refused_when_grow_fails_near_full();
    test_str_map_update_survives_failed_grow_near_full();
    test_merge_map_update_survives_failed_grow_near_full();
    test_str_map_foreach_visits_live_entries_once();
    test_str_map_init_huge_cap();
    test_merge_map_init_huge_cap();
    printf("All tok_map tests passed.\n");
    return 0;
}
