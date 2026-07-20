#include "engine/forward.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

/* Cycle 1: CPU-only API / struct smoke. No mlx graph execution. */

static void test_structs_zero_init(void) {
    fwd_moe_weights_t mw;
    fwd_moe_params_t p;
    memset(&mw, 0, sizeof(mw));
    memset(&p, 0, sizeof(p));
    assert(sizeof(fwd_moe_weights_t) > 0);
    assert(sizeof(fwd_moe_params_t) > 0);
    assert(!mw.has_shared);
    assert(!mw.has_shared_expert_gate);
    assert(p.num_experts == 0);
    assert(p.top_k == 0);
    assert(!p.norm_topk_prob);
    assert(mw.router.weight.ctx == NULL);
    assert(mw.switch_gate.weight.ctx == NULL);
}

static void test_null_args_return_error(void) {
    mlx_array dummy = mlx_array_new();
    mlx_stream s = {.ctx = NULL};

    assert(fwd_array_contiguous(NULL, dummy, s) == -1);
    assert(fwd_moe_route_softmax(NULL, &dummy, dummy, 1, false, s) == -1);
    assert(fwd_moe_route_softmax(&dummy, NULL, dummy, 1, false, s) == -1);
    assert(fwd_moe_route_softmax(&dummy, &dummy, (mlx_array){.ctx = NULL},
                                 1, false, s) == -1);
    assert(fwd_moe_route_softmax(&dummy, &dummy, dummy, 0, false, s) == -1);

    assert(fwd_gather_expert_linear(NULL, dummy, dummy, NULL,
                                    (mlx_array){.ctx = NULL}, false,
                                    NULL, s) == -1);
    assert(fwd_switch_glu(NULL, dummy, dummy, NULL, NULL, NULL, NULL, s) == -1);
    assert(fwd_moe(NULL, dummy, NULL, NULL, NULL, s) == -1);

    mlx_array_free(dummy);
}

int main(void) {
    test_structs_zero_init();
    printf("  test_structs_zero_init: passed\n");
    test_null_args_return_error();
    printf("  test_null_args_return_error: passed\n");
    printf("all test_fwd_moe_api tests passed\n");
    return 0;
}
