#include "http/response.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char *http_status_reason(int status) {
    switch (status) {
    case 200: return "OK";
    case 204: return "No Content";
    case 400: return "Bad Request";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 413: return "Payload Too Large";
    case 500: return "Internal Server Error";
    case 501: return "Not Implemented";
    case 503: return "Service Unavailable";
    default:  return "Unknown";
    }
}

char *http_build_response(int status, const char *content_type, const char *body,
                          size_t body_len, bool keep_alive, size_t *out_len) {
    const char *reason = http_status_reason(status);
    const char *conn = keep_alive ? "keep-alive" : "close";

    char head[512];
    int head_len = snprintf(head, sizeof(head),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %zu\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Connection: %s\r\n"
        "\r\n",
        status, reason, content_type, body_len, conn);

    if (head_len < 0 || (size_t)head_len >= sizeof(head))
        return NULL;

    size_t total = (size_t)head_len + body_len;
    char *buf = malloc(total + 1);
    if (!buf) return NULL;

    memcpy(buf, head, (size_t)head_len);
    if (body && body_len > 0)
        memcpy(buf + head_len, body, body_len);
    buf[total] = '\0';

    if (out_len) *out_len = total;
    return buf;
}

char *http_build_sse_head(size_t *out_len) {
    const char *head =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Connection: close\r\n"
        "\r\n";
    size_t len = strlen(head);
    char *buf = malloc(len + 1);
    if (!buf) return NULL;
    memcpy(buf, head, len + 1);
    if (out_len) *out_len = len;
    return buf;
}
