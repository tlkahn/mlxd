#include "model/detok.h"
#include "model/tokenizer.h"

#include <stdlib.h>
#include <string.h>

struct detok {
    const tokenizer_t *tok;
    char  *buf;
    size_t len;
    size_t cap;
};

static size_t utf8_char_len(unsigned char lead) {
    if (lead < 0x80) return 1;
    if ((lead & 0xE0) == 0xC0) return 2;
    if ((lead & 0xF0) == 0xE0) return 3;
    if ((lead & 0xF8) == 0xF0) return 4;
    return 1; /* invalid lead / bare continuation: pass through */
}

static size_t utf8_complete_prefix(const char *buf, size_t len) {
    size_t i = 0;
    while (i < len) {
        size_t cl = utf8_char_len((unsigned char)buf[i]);
        if (i + cl > len) break;
        i += cl;
    }
    return i;
}

static int emit(detok_t *d, size_t boundary, char **out, size_t *out_len) {
    char *o = malloc(boundary + 1);
    if (!o) return -1;
    if (boundary > 0) memcpy(o, d->buf, boundary);
    o[boundary] = '\0';
    *out = o;
    *out_len = boundary;
    size_t remain = d->len - boundary;
    if (remain > 0) memmove(d->buf, d->buf + boundary, remain);
    d->len = remain;
    return 0;
}

detok_t *detok_create(const tokenizer_t *tok) {
    if (!tok || tokenizer_type(tok) == TOKENIZER_WORDPIECE)
        return NULL;
    detok_t *d = calloc(1, sizeof(*d));
    if (!d) return NULL;
    d->tok = tok;
    return d;
}

int detok_feed(detok_t *d, int32_t id, char **out, size_t *out_len) {
    char *s = tokenizer_decode(d->tok, &id, 1);
    if (!s) return -1;
    size_t slen = strlen(s);
    if (slen > 0) {
        size_t need = d->len + slen;
        if (need > d->cap) {
            size_t new_cap = d->cap ? d->cap : 64;
            while (new_cap < need) new_cap *= 2;
            char *tmp = realloc(d->buf, new_cap);
            if (!tmp) { free(s); return -1; }
            d->buf = tmp;
            d->cap = new_cap;
        }
        memcpy(d->buf + d->len, s, slen);
        d->len += slen;
    }
    free(s);
    size_t boundary = utf8_complete_prefix(d->buf, d->len);
    return emit(d, boundary, out, out_len);
}

int detok_flush(detok_t *d, char **out, size_t *out_len) {
    return emit(d, d->len, out, out_len);
}

void detok_free(detok_t *d) {
    if (!d) return;
    free(d->buf);
    free(d);
}
