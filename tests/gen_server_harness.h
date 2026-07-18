#ifndef MLXD_TEST_GEN_SERVER_HARNESS_H
#define MLXD_TEST_GEN_SERVER_HARNESS_H

#include "http/server.h"
#include "model/tokenizer.h"
#include "http_client.h"

#include <assert.h>
#include <pthread.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/time.h>

#ifndef MLXD_FIXTURES_DIR
#error "MLXD_FIXTURES_DIR must be defined"
#endif

#define TRIVIAL_TMPL "{{ messages[0].content }}"

typedef struct {
    engine_t       eng;
    tokenizer_t   *tok;
    http_server_t *srv;
    pthread_t      th;
    int            port;
} gen_fixture_t;

static void *gen_server_thread(void *arg) {
    http_server_start((http_server_t *)arg);
    return NULL;
}

static gen_fixture_t gen_fixture_up(bool load_model, bool set_tokenizer,
                                    const char *chat_template,
                                    uint64_t drain_deadline_ms) {
    gen_fixture_t f = {0};
    engine_init(&f.eng);

    if (load_model) {
        engine_cmd_t *cmd = calloc(1, sizeof(*cmd));
        assert(cmd != NULL);
        cmd->tag = CMD_LOAD;
        cmd->load.model_path = strdup(MLXD_STUB_MODEL_PATH);
        engine_post(&f.eng, cmd);
        usleep(10000);
    }

    f.tok = set_tokenizer
        ? tokenizer_load(MLXD_FIXTURES_DIR "/gpt2/tokenizer.json")
        : NULL;

    http_server_config_t cfg = {
        .port = 0,
        .engine = &f.eng,
        .tokenizer = f.tok,
        .chat_template = chat_template,
        .model_id = "gpt2",
        .drain_deadline_ms = drain_deadline_ms,
    };
    f.srv = http_server_create(&cfg);
    assert(f.srv != NULL);
    f.port = http_server_port(f.srv);
    assert(f.port > 0);
    int rc = pthread_create(&f.th, NULL, gen_server_thread, f.srv);
    assert(rc == 0);
    for (int i = 0; i < 500; i++) {
        int fd = http_client_connect("127.0.0.1", f.port);
        if (fd >= 0) { close(fd); break; }
        usleep(1000);
        assert(i < 499);
    }
    return f;
}

static void gen_fixture_down(gen_fixture_t *f) {
    http_server_stop(f->srv);
    pthread_join(f->th, NULL);
    http_server_destroy(f->srv);
    engine_destroy(&f->eng);
    if (f->tok) tokenizer_free(f->tok);
}

static uint64_t now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000 + (uint64_t)tv.tv_usec / 1000;
}

/* Wait until pending unread bytes on fd are positive and stable across
   consecutive polls, meaning the server's write path is saturated. */
static void wait_for_socket_saturation(int fd) {
    int prev = -1, stable = 0;
    for (int i = 0; i < 500; i++) {
        int avail = 0;
        assert(ioctl(fd, FIONREAD, &avail) == 0);
        if (avail > 0 && avail == prev) {
            if (++stable >= 5) return;
        } else {
            stable = 0;
        }
        prev = avail;
        usleep(10000);
    }
    assert(!"socket never saturated");
}

static int sse_connect_and_post(int port, const char *path, const char *body,
                                char *hdrbuf, size_t hdrcap) {
    int fd = http_client_connect("127.0.0.1", port);
    assert(fd >= 0);

    char raw[4096];
    snprintf(raw, sizeof(raw),
        "POST %s HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %zu\r\n"
        "\r\n%s", path, strlen(body), body);
    assert(http_client_send_all(fd, raw, strlen(raw)) == 0);

    int rc = http_client_recv_headers(fd, hdrbuf, hdrcap);
    assert(rc > 0);
    return fd;
}

#endif
