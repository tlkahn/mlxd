#include "http_client.h"

#include <assert.h>
#include <stdio.h>
#include <sys/socket.h>
#include <unistd.h>

static void test_sse_event_boundary(void) {
    int fds[2];
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0);

    const char *stream = "data: a\n\ndata: b\n\n";
    assert(http_client_send_all(fds[1], stream, strlen(stream)) == 0);
    close(fds[1]);

    char buf[256];
    int n1 = http_client_recv_sse_event(fds[0], buf, sizeof(buf));
    assert(n1 == 9);
    assert(memcmp(buf, "data: a\n\n", 9) == 0);

    int n2 = http_client_recv_sse_event(fds[0], buf, sizeof(buf));
    assert(n2 == 9);
    assert(memcmp(buf, "data: b\n\n", 9) == 0);

    close(fds[0]);
}

int main(void) {
    test_sse_event_boundary();
    printf("test_http_client: all tests passed\n");
    return 0;
}
