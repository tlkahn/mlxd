#ifndef MLXD_TEST_HTTP_CLIENT_H
#define MLXD_TEST_HTTP_CLIENT_H

#include <arpa/inet.h>
#include <assert.h>
#include <ctype.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <unistd.h>

static const char *hc_strcasestr(const char *haystack, const char *needle) {
    size_t nlen = strlen(needle);
    if (nlen == 0) return haystack;
    for (; *haystack; haystack++) {
        if (strncasecmp(haystack, needle, nlen) == 0)
            return haystack;
    }
    return NULL;
}

typedef struct {
    int    status;
    char   headers[8192];
    char  *body;
    size_t body_len;
    bool   keep_alive;
} http_client_response_t;

static int http_client_connect(const char *host, int port) {
    static int sigpipe_done = 0;
    if (!sigpipe_done) {
        signal(SIGPIPE, SIG_IGN);
        sigpipe_done = 1;
    }

    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;

    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    inet_pton(AF_INET, host, &addr.sin_addr);

    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        close(fd);
        return -1;
    }
    return fd;
}

static int http_client_send_all(int fd, const char *data, size_t len) {
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = write(fd, data + sent, len - sent);
        if (n <= 0) return -1;
        sent += (size_t)n;
    }
    return 0;
}

static int http_client_recv_response(int fd, http_client_response_t *out) {
    memset(out, 0, sizeof(*out));

    char buf[65536];
    size_t total = 0;
    char *hdr_end = NULL;

    while (!hdr_end && total < sizeof(buf) - 1) {
        ssize_t n = read(fd, buf + total, sizeof(buf) - 1 - total);
        if (n <= 0) {
            if (total == 0) return -1;
            break;
        }
        total += (size_t)n;
        buf[total] = '\0';
        hdr_end = strstr(buf, "\r\n\r\n");
    }

    if (!hdr_end) return -1;

    size_t hdr_len = (size_t)(hdr_end - buf);
    if (hdr_len >= sizeof(out->headers)) hdr_len = sizeof(out->headers) - 1;
    memcpy(out->headers, buf, hdr_len);
    out->headers[hdr_len] = '\0';

    sscanf(buf, "HTTP/1.1 %d", &out->status);

    out->keep_alive = true;
    const char *conn = hc_strcasestr(out->headers, "Connection:");
    if (conn) {
        conn += 11;
        while (*conn == ' ') conn++;
        if (strncasecmp(conn, "close", 5) == 0)
            out->keep_alive = false;
    }

    size_t content_length = 0;
    const char *cl = hc_strcasestr(out->headers, "Content-Length:");
    if (cl) {
        cl += 15;
        while (*cl == ' ') cl++;
        content_length = (size_t)atol(cl);
    }

    char *body_start = hdr_end + 4;
    size_t body_have = total - (size_t)(body_start - buf);

    if (content_length > 0) {
        out->body = malloc(content_length + 1);
        if (!out->body) return -1;

        size_t copy = body_have < content_length ? body_have : content_length;
        memcpy(out->body, body_start, copy);
        size_t got = copy;

        while (got < content_length) {
            ssize_t n = read(fd, out->body + got, content_length - got);
            if (n <= 0) break;
            got += (size_t)n;
        }
        out->body_len = got;
        out->body[got] = '\0';
    } else {
        out->body = NULL;
        out->body_len = 0;
    }

    return 0;
}

static int http_client_request(const char *host, int port, const char *raw,
                               http_client_response_t *out) {
    int fd = http_client_connect(host, port);
    if (fd < 0) return -1;

    if (http_client_send_all(fd, raw, strlen(raw)) < 0) {
        close(fd);
        return -1;
    }

    int rc = http_client_recv_response(fd, out);
    close(fd);
    return rc;
}

static bool http_client_header(const http_client_response_t *r, const char *name,
                               char *out, size_t cap) {
    size_t nlen = strlen(name);
    const char *p = r->headers;
    while (*p) {
        if (strncasecmp(p, name, nlen) == 0 && p[nlen] == ':') {
            const char *v = p + nlen + 1;
            while (*v == ' ') v++;
            const char *end = strstr(v, "\r\n");
            if (!end) end = v + strlen(v);
            size_t vlen = (size_t)(end - v);
            if (vlen >= cap) vlen = cap - 1;
            memcpy(out, v, vlen);
            out[vlen] = '\0';
            return true;
        }
        const char *nl = strstr(p, "\r\n");
        if (!nl) break;
        p = nl + 2;
    }
    return false;
}

static void http_client_response_free(http_client_response_t *r) {
    free(r->body);
    r->body = NULL;
    r->body_len = 0;
}

static int http_client_recv_headers(int fd, char *out, size_t cap) {
    size_t total = 0;
    while (total < cap - 1) {
        ssize_t n = read(fd, out + total, 1);
        if (n <= 0) return -1;
        total += 1;
        out[total] = '\0';
        if (total >= 4 && memcmp(out + total - 4, "\r\n\r\n", 4) == 0)
            return (int)total;
    }
    return -1;
}

static int http_client_recv_sse_event(int fd, char *out, size_t cap) {
    size_t total = 0;
    while (total < cap - 1) {
        ssize_t n = read(fd, out + total, 1);
        if (n <= 0) return -1;
        total += 1;
        out[total] = '\0';
        if (total >= 2 && memcmp(out + total - 2, "\n\n", 2) == 0)
            return (int)total;
    }
    return -1;
}

#endif
