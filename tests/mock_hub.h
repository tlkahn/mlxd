#ifndef MOCK_HUB_H
#define MOCK_HUB_H

#include <strings.h>
#include <arpa/inet.h>
#include <assert.h>
#include <netinet/in.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#define MOCK_HUB_MAX_ROUTES 32
#define MOCK_HUB_BUF_SIZE  8192

typedef struct {
    const char *method;
    const char *path_prefix;
    int         status;
    const char *body;
    size_t      body_len;
    int         honor_range;
} mock_route_t;

typedef struct {
    char auth[256];
    char range[256];
    char path[1024];
    char ua[256];
} mock_recorded_t;

typedef struct {
    int              listen_fd;
    int              port;
    pthread_t        thread;
    atomic_int       stop;
    mock_route_t     routes[MOCK_HUB_MAX_ROUTES];
    int              route_count;
    pthread_mutex_t  mu;
    mock_recorded_t  recorded;
} mock_hub_t;

static void mock_hub_init(mock_hub_t *h) {
    memset(h, 0, sizeof(*h));
    h->listen_fd = -1;
    pthread_mutex_init(&h->mu, NULL);
}

static void mock_hub_add(mock_hub_t *h, const char *method, const char *path_prefix,
                         int status, const char *body, size_t body_len, int honor_range) {
    assert(h->route_count < MOCK_HUB_MAX_ROUTES);
    mock_route_t *r = &h->routes[h->route_count++];
    r->method = method;
    r->path_prefix = path_prefix;
    r->status = status;
    r->body = body;
    r->body_len = body_len;
    r->honor_range = honor_range;
}

static void mock_hub_send_all(int fd, const char *buf, size_t len) {
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = write(fd, buf + sent, len - sent);
        if (n <= 0) break;
        sent += (size_t)n;
    }
}

static void mock_hub_handle(mock_hub_t *h, int client_fd) {
    char buf[MOCK_HUB_BUF_SIZE];
    ssize_t n = read(client_fd, buf, sizeof(buf) - 1);
    if (n <= 0) {
        close(client_fd);
        return;
    }
    buf[n] = '\0';

    char method[16] = {0};
    char path[1024] = {0};
    sscanf(buf, "%15s %1023s", method, path);

    char auth[256] = {0};
    char range_hdr[256] = {0};
    char ua[256] = {0};
    const char *p = buf;
    while ((p = strstr(p, "\r\n")) != NULL) {
        p += 2;
        if (*p == '\r' || *p == '\0') break;
        if (strncasecmp(p, "Authorization:", 14) == 0) {
            const char *v = p + 14;
            while (*v == ' ') v++;
            char *end = strstr(v, "\r\n");
            size_t vlen = end ? (size_t)(end - v) : strlen(v);
            if (vlen >= sizeof(auth)) vlen = sizeof(auth) - 1;
            memcpy(auth, v, vlen);
            auth[vlen] = '\0';
        }
        if (strncasecmp(p, "Range:", 6) == 0) {
            const char *v = p + 6;
            while (*v == ' ') v++;
            char *end = strstr(v, "\r\n");
            size_t vlen = end ? (size_t)(end - v) : strlen(v);
            if (vlen >= sizeof(range_hdr)) vlen = sizeof(range_hdr) - 1;
            memcpy(range_hdr, v, vlen);
            range_hdr[vlen] = '\0';
        }
        if (strncasecmp(p, "User-Agent:", 11) == 0) {
            const char *v = p + 11;
            while (*v == ' ') v++;
            char *end = strstr(v, "\r\n");
            size_t vlen = end ? (size_t)(end - v) : strlen(v);
            if (vlen >= sizeof(ua)) vlen = sizeof(ua) - 1;
            memcpy(ua, v, vlen);
            ua[vlen] = '\0';
        }
    }

    pthread_mutex_lock(&h->mu);
    snprintf(h->recorded.auth, sizeof(h->recorded.auth), "%s", auth);
    snprintf(h->recorded.range, sizeof(h->recorded.range), "%s", range_hdr);
    snprintf(h->recorded.path, sizeof(h->recorded.path), "%s", path);
    snprintf(h->recorded.ua, sizeof(h->recorded.ua), "%s", ua);
    pthread_mutex_unlock(&h->mu);

    mock_route_t *match = NULL;
    for (int i = 0; i < h->route_count; i++) {
        if (strcmp(method, h->routes[i].method) != 0)
            continue;
        if (strncmp(path, h->routes[i].path_prefix, strlen(h->routes[i].path_prefix)) != 0)
            continue;
        match = &h->routes[i];
        break;
    }

    if (!match) {
        const char *resp = "HTTP/1.1 404 Not Found\r\nConnection: close\r\nContent-Length: 0\r\n\r\n";
        mock_hub_send_all(client_fd, resp, strlen(resp));
        close(client_fd);
        return;
    }

    size_t offset = 0;
    size_t send_len = match->body_len;
    int resp_status = match->status;

    if (match->honor_range && range_hdr[0]) {
        long start = 0;
        if (sscanf(range_hdr, "bytes=%ld-", &start) == 1 && start >= 0) {
            if ((size_t)start >= match->body_len) {
                char hdr[256];
                int hlen = snprintf(hdr, sizeof(hdr),
                    "HTTP/1.1 416 Range Not Satisfiable\r\n"
                    "Connection: close\r\nContent-Length: 0\r\n\r\n");
                mock_hub_send_all(client_fd, hdr, (size_t)hlen);
                close(client_fd);
                return;
            }
            offset = (size_t)start;
            send_len = match->body_len - offset;
            resp_status = 206;
        }
    }

    char hdr[512];
    int hlen;
    if (resp_status == 206) {
        hlen = snprintf(hdr, sizeof(hdr),
            "HTTP/1.1 206 Partial Content\r\n"
            "Connection: close\r\n"
            "Content-Length: %zu\r\n"
            "Content-Range: bytes %zu-%zu/%zu\r\n\r\n",
            send_len, offset, offset + send_len - 1, match->body_len);
    } else {
        const char *status_text = "OK";
        if (resp_status == 401) status_text = "Unauthorized";
        else if (resp_status == 403) status_text = "Forbidden";
        else if (resp_status == 404) status_text = "Not Found";
        hlen = snprintf(hdr, sizeof(hdr),
            "HTTP/1.1 %d %s\r\n"
            "Connection: close\r\n"
            "Content-Length: %zu\r\n\r\n",
            resp_status, status_text, send_len);
    }
    mock_hub_send_all(client_fd, hdr, (size_t)hlen);
    if (send_len > 0)
        mock_hub_send_all(client_fd, match->body + offset, send_len);
    close(client_fd);
}

static void *mock_hub_thread(void *arg) {
    mock_hub_t *h = (mock_hub_t *)arg;
    while (!atomic_load(&h->stop)) {
        int client = accept(h->listen_fd, NULL, NULL);
        if (client < 0) continue;
        if (atomic_load(&h->stop)) {
            close(client);
            break;
        }
        mock_hub_handle(h, client);
    }
    return NULL;
}

static int mock_hub_start(mock_hub_t *h) {
    static int sigpipe_done = 0;
    if (!sigpipe_done) {
        signal(SIGPIPE, SIG_IGN);
        sigpipe_done = 1;
    }

    h->listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (h->listen_fd < 0) return -1;

    int opt = 1;
    setsockopt(h->listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(0x7f000001u);
    addr.sin_port = 0;

    if (bind(h->listen_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        close(h->listen_fd);
        return -1;
    }
    if (listen(h->listen_fd, 8) < 0) {
        close(h->listen_fd);
        return -1;
    }

    struct sockaddr_in bound;
    socklen_t blen = sizeof(bound);
    getsockname(h->listen_fd, (struct sockaddr *)&bound, &blen);
    h->port = ntohs(bound.sin_port);

    atomic_store(&h->stop, 0);
    if (pthread_create(&h->thread, NULL, mock_hub_thread, h) != 0) {
        close(h->listen_fd);
        return -1;
    }
    return 0;
}

static void mock_hub_base_url(mock_hub_t *h, char *buf, size_t bufsz) {
    snprintf(buf, bufsz, "http://127.0.0.1:%d", h->port);
}

static void mock_hub_stop(mock_hub_t *h) {
    atomic_store(&h->stop, 1);
    int wake = socket(AF_INET, SOCK_STREAM, 0);
    if (wake >= 0) {
        struct sockaddr_in addr = {0};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(0x7f000001u);
        addr.sin_port = htons((uint16_t)h->port);
        connect(wake, (struct sockaddr *)&addr, sizeof(addr));
        close(wake);
    }
    pthread_join(h->thread, NULL);
    close(h->listen_fd);
    h->listen_fd = -1;
    pthread_mutex_destroy(&h->mu);
}

static void mock_hub_get_recorded(mock_hub_t *h, mock_recorded_t *out) {
    pthread_mutex_lock(&h->mu);
    *out = h->recorded;
    pthread_mutex_unlock(&h->mu);
}

#endif
