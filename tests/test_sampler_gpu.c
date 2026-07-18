/* GPU tests for the Stage C sampler pipeline (src/engine/sampler.{h,c}).
 * Crafted logits only - no model load. Run via `make test-gpu`. */

#include "engine/sampler.h"
#include "mlxbridge/mlxbridge.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#define VOCAB 8

static mlx_array logits_1xv(const float *vals) {
    int shape[] = {1, VOCAB};
    return mlx_array_new_data(vals, shape, 2, MLX_FLOAT32);
}

/* argmax lives at index 2 */
static const float kLogits[VOCAB] = {0.1f, -2.0f, 3.5f, 1.0f, 0.0f, 2.9f, -1.0f, 0.5f};

/* ---- Cycle 1: temperature 0 hits the greedy limit ---- */

static void test_greedy_limit(void) {
    mlx_stream s = mlxbridge_gpu_stream();
    mlx_array logits = logits_1xv(kLogits);
    mlx_array nokey = mlx_array_new();

    sampling_params_t sp = SAMPLING_PARAMS_DEFAULT;
    sp.temperature = 0.0f;

    mlx_array tok = mlx_array_new();
    assert(sampler_sample_lazy(logits, &sp, nokey, false, s, &tok, NULL) == 0);
    int32_t id = -1;
    assert(mlxbridge_item_int32(&id, tok) == 0);
    assert(id == 2);

    mlx_array_free(tok);
    mlx_array_free(nokey);
    mlx_array_free(logits);
    mlx_stream_free(s);
}

/* ---- Cycle 2: top_k filter masks all but the k largest with -inf ---- */

/* Eval `arr` and copy its VOCAB floats into out. */
static void readback_1xv(mlx_array arr, float *out) {
    assert(mlx_array_eval(arr) == 0);
    const float *data = mlx_array_data_float32(arr);
    assert(data != NULL);
    for (int i = 0; i < VOCAB; i++)
        out[i] = data[i];
}

static void test_top_k_filter(void) {
    mlx_stream s = mlxbridge_gpu_stream();
    mlx_array logits = logits_1xv(kLogits);
    float out[VOCAB];

    /* k=3 keeps indices 2 (3.5), 5 (2.9), 3 (1.0); rest -inf */
    mlx_array filtered = mlx_array_new();
    assert(sampler_apply_top_k(logits, 3, s, &filtered) == 0);
    readback_1xv(filtered, out);
    for (int i = 0; i < VOCAB; i++) {
        if (i == 2 || i == 5 || i == 3)
            assert(out[i] == kLogits[i]);
        else
            assert(isinf(out[i]) && out[i] < 0);
    }
    mlx_array_free(filtered);

    /* k=-1 (disabled) and k >= vocab leave logits unchanged */
    int passthrough_ks[] = {-1, VOCAB, VOCAB + 5};
    for (size_t t = 0; t < sizeof(passthrough_ks) / sizeof(*passthrough_ks); t++) {
        mlx_array same = mlx_array_new();
        assert(sampler_apply_top_k(logits, passthrough_ks[t], s, &same) == 0);
        readback_1xv(same, out);
        for (int i = 0; i < VOCAB; i++)
            assert(out[i] == kLogits[i]);
        mlx_array_free(same);
    }

    mlx_array_free(logits);
    mlx_stream_free(s);
}

/* ---- Cycle 3: top_k=1 at high temperature still equals greedy ---- */

static void test_top_k1_equals_greedy(void) {
    mlx_stream s = mlxbridge_gpu_stream();
    mlx_array logits = logits_1xv(kLogits);
    mlx_array nokey = mlx_array_new();

    sampling_params_t sp = SAMPLING_PARAMS_DEFAULT;
    sp.temperature = 5.0f;
    sp.top_k = 1;

    for (int i = 0; i < 16; i++) {
        mlx_array tok = mlx_array_new();
        assert(sampler_sample_lazy(logits, &sp, nokey, false, s, &tok, NULL) == 0);
        int32_t id = -1;
        assert(mlxbridge_item_int32(&id, tok) == 0);
        assert(id == 2);
        mlx_array_free(tok);
    }

    mlx_array_free(nokey);
    mlx_array_free(logits);
    mlx_stream_free(s);
}

/* ---- Cycle 4: top_p nucleus filter ---- */

static void test_top_p_filter(void) {
    mlx_stream s = mlxbridge_gpu_stream();

    /* unsorted probs {.15, .5, .05, .3}; nucleus for p=0.7 is {.5, .3} = idx 1, 3 */
    const float probs[4] = {0.15f, 0.5f, 0.05f, 0.3f};
    float vals[4];
    for (int i = 0; i < 4; i++)
        vals[i] = logf(probs[i]);
    int shape[] = {1, 4};
    mlx_array logits = mlx_array_new_data(vals, shape, 2, MLX_FLOAT32);
    float out[4];

    mlx_array filtered = mlx_array_new();
    assert(sampler_apply_top_p(logits, 0.7f, s, &filtered) == 0);
    assert(mlx_array_eval(filtered) == 0);
    const float *data = mlx_array_data_float32(filtered);
    assert(data != NULL);
    for (int i = 0; i < 4; i++)
        out[i] = data[i];
    for (int i = 0; i < 4; i++) {
        if (i == 1 || i == 3)
            assert(fabsf(out[i] - vals[i]) < 1e-6f);
        else
            assert(isinf(out[i]) && out[i] < 0);
    }
    mlx_array_free(filtered);

    /* top_p = 1.0 (edge case from the issue): logits unchanged */
    mlx_array same = mlx_array_new();
    assert(sampler_apply_top_p(logits, 1.0f, s, &same) == 0);
    assert(mlx_array_eval(same) == 0);
    data = mlx_array_data_float32(same);
    assert(data != NULL);
    for (int i = 0; i < 4; i++)
        assert(fabsf(data[i] - vals[i]) < 1e-6f);
    mlx_array_free(same);

    mlx_array_free(logits);
    mlx_stream_free(s);
}

/* ---- Review fix D: top_p = 0 degenerates to greedy, not NaN ---- */

static void test_top_p_zero_greedy(void) {
    mlx_stream s = mlxbridge_gpu_stream();
    mlx_array logits = logits_1xv(kLogits);
    float out[VOCAB];

    /* Filter level: p=0 keeps only the max logit (idx 2), rest -inf */
    mlx_array filtered = mlx_array_new();
    assert(sampler_apply_top_p(logits, 0.0f, s, &filtered) == 0);
    readback_1xv(filtered, out);
    for (int i = 0; i < VOCAB; i++) {
        if (i == 2)
            assert(out[i] == kLogits[i]);
        else
            assert(isinf(out[i]) && out[i] < 0);
    }
    mlx_array_free(filtered);

    /* Pipeline level: high temperature + top_p=0 always draws the argmax
     * (an all -inf row would make categorical return NaN/garbage) */
    mlx_array nokey = mlx_array_new();
    sampling_params_t sp = SAMPLING_PARAMS_DEFAULT;
    sp.temperature = 5.0f;
    sp.top_p = 0.0f;
    for (int i = 0; i < 16; i++) {
        mlx_array tok = mlx_array_new();
        assert(sampler_sample_lazy(logits, &sp, nokey, false, s, &tok, NULL) == 0);
        int32_t id = -1;
        assert(mlxbridge_item_int32(&id, tok) == 0);
        assert(id == 2);
        mlx_array_free(tok);
    }
    mlx_array_free(nokey);

    mlx_array_free(logits);
    mlx_stream_free(s);
}

/* ---- Cycle 5: min_p filter (no oracle, pure logit-space) ---- */

static void test_min_p_filter(void) {
    mlx_stream s = mlxbridge_gpu_stream();
    mlx_array logits = logits_1xv(kLogits);
    float out[VOCAB];

    /* kLogits max is 3.5 at idx 2. min_p=0.5 threshold = 3.5 + log(0.5) ~= 2.807
     * Only idx 2 (3.5) and idx 5 (2.9) survive. */
    mlx_array filtered = mlx_array_new();
    assert(sampler_apply_min_p(logits, 0.5f, s, &filtered) == 0);
    readback_1xv(filtered, out);
    for (int i = 0; i < VOCAB; i++) {
        if (i == 2 || i == 5)
            assert(out[i] == kLogits[i]);
        else
            assert(isinf(out[i]) && out[i] < 0);
    }
    mlx_array_free(filtered);

    /* min_p = 0 (disabled): logits unchanged */
    mlx_array same = mlx_array_new();
    assert(sampler_apply_min_p(logits, 0.0f, s, &same) == 0);
    readback_1xv(same, out);
    for (int i = 0; i < VOCAB; i++)
        assert(out[i] == kLogits[i]);
    mlx_array_free(same);

    mlx_array_free(logits);
    mlx_stream_free(s);
}

/* ---- Cycle 6: seeded reproducibility + key split ---- */

static void test_seeded_reproducibility(void) {
    mlx_stream s = mlxbridge_gpu_stream();

    /* Near-uniform logits so categorical actually varies */
    const float uniform[VOCAB] = {0.0f, 0.01f, -0.01f, 0.02f, -0.02f, 0.03f, -0.03f, 0.015f};
    mlx_array logits = logits_1xv(uniform);

    sampling_params_t sp = SAMPLING_PARAMS_DEFAULT;
    sp.temperature = 1.0f;

    /* Two runs with same seed (42) must produce identical 16-draw sequences */
    int32_t seq_a[16], seq_b[16];
    for (int run = 0; run < 2; run++) {
        int32_t *seq = (run == 0) ? seq_a : seq_b;
        sampler_key_t key;
        assert(sampler_key_init(&key, 42) == 0);
        for (int i = 0; i < 16; i++) {
            mlx_array subkey = mlx_array_new();
            assert(sampler_key_next(&key, &subkey, s) == 0);
            mlx_array tok = mlx_array_new();
            assert(sampler_sample_lazy(logits, &sp, subkey, false, s, &tok, NULL) == 0);
            assert(mlxbridge_item_int32(&seq[i], tok) == 0);
            mlx_array_free(tok);
            mlx_array_free(subkey);
        }
        sampler_key_free(&key);
    }
    for (int i = 0; i < 16; i++)
        assert(seq_a[i] == seq_b[i]);

    /* Different seed (7) produces a different sequence */
    int32_t seq_c[16];
    {
        sampler_key_t key;
        assert(sampler_key_init(&key, 7) == 0);
        for (int i = 0; i < 16; i++) {
            mlx_array subkey = mlx_array_new();
            assert(sampler_key_next(&key, &subkey, s) == 0);
            mlx_array tok = mlx_array_new();
            assert(sampler_sample_lazy(logits, &sp, subkey, false, s, &tok, NULL) == 0);
            assert(mlxbridge_item_int32(&seq_c[i], tok) == 0);
            mlx_array_free(tok);
            mlx_array_free(subkey);
        }
        sampler_key_free(&key);
    }
    bool differs = false;
    for (int i = 0; i < 16; i++) {
        if (seq_a[i] != seq_c[i]) { differs = true; break; }
    }
    assert(differs);

    /* Consecutive subkeys from one key produce different draws on uniform logits */
    {
        sampler_key_t key;
        assert(sampler_key_init(&key, 99) == 0);
        bool all_same = true;
        int32_t first = -1;
        for (int i = 0; i < 16; i++) {
            mlx_array subkey = mlx_array_new();
            assert(sampler_key_next(&key, &subkey, s) == 0);
            mlx_array tok = mlx_array_new();
            assert(sampler_sample_lazy(logits, &sp, subkey, false, s, &tok, NULL) == 0);
            int32_t id;
            assert(mlxbridge_item_int32(&id, tok) == 0);
            if (i == 0) first = id;
            else if (id != first) all_same = false;
            mlx_array_free(tok);
            mlx_array_free(subkey);
        }
        sampler_key_free(&key);
        assert(!all_same);
    }

    mlx_array_free(logits);
    mlx_stream_free(s);
}

/* ---- Cycle 10: lazy logprob computation ---- */

static void test_logprob_computation(void) {
    mlx_stream s = mlxbridge_gpu_stream();

    /* Reference logprob from RAW (pre-temperature) logits.
     * log(softmax(kLogits)[2]) = kLogits[2] - logsumexp(kLogits). */
    float maxr = kLogits[0];
    for (int i = 1; i < VOCAB; i++)
        if (kLogits[i] > maxr) maxr = kLogits[i];
    double sum_exp = 0;
    for (int i = 0; i < VOCAB; i++)
        sum_exp += exp((double)(kLogits[i] - maxr));
    double logsumexp = (double)maxr + log(sum_exp);
    float expected_lp = (float)((double)kLogits[2] - logsumexp);

    mlx_array logits = logits_1xv(kLogits);
    mlx_array nokey = mlx_array_new();

    /* Non-greedy path: temperature=0.7, top_k=1 forces idx 2 */
    {
        sampling_params_t sp = SAMPLING_PARAMS_DEFAULT;
        sp.temperature = 0.7f;
        sp.top_k = 1;

        mlx_array tok = mlx_array_new();
        mlx_array logprob_arr = mlx_array_new();
        assert(sampler_sample_lazy(logits, &sp, nokey, true, s, &tok, &logprob_arr) == 0);

        int32_t id;
        assert(mlxbridge_item_int32(&id, tok) == 0);
        assert(id == 2);

        float lp;
        assert(mlxbridge_item_float32(&lp, logprob_arr) == 0);
        assert(fabsf(lp - expected_lp) < 1e-4f);

        mlx_array_free(logprob_arr);
        mlx_array_free(tok);
    }

    /* Greedy path: temperature=0 */
    {
        sampling_params_t sp = SAMPLING_PARAMS_DEFAULT;
        sp.temperature = 0.0f;

        mlx_array tok = mlx_array_new();
        mlx_array logprob_arr = mlx_array_new();
        assert(sampler_sample_lazy(logits, &sp, nokey, true, s, &tok, &logprob_arr) == 0);

        int32_t id;
        assert(mlxbridge_item_int32(&id, tok) == 0);
        assert(id == 2);

        float lp;
        assert(mlxbridge_item_float32(&lp, logprob_arr) == 0);
        assert(fabsf(lp - expected_lp) < 1e-4f);

        mlx_array_free(logprob_arr);
        mlx_array_free(tok);
    }

    /* want_logprob=false: logprob_arr untouched */
    {
        sampling_params_t sp = SAMPLING_PARAMS_DEFAULT;
        sp.temperature = 0.7f;
        sp.top_k = 1;

        mlx_array tok2 = mlx_array_new();
        mlx_array logprob2 = mlx_array_new();
        assert(sampler_sample_lazy(logits, &sp, nokey, false, s, &tok2, &logprob2) == 0);

        mlx_array_free(logprob2);
        mlx_array_free(tok2);
    }

    mlx_array_free(nokey);
    mlx_array_free(logits);
    mlx_stream_free(s);
}

/* ---- Cycle 2 fix: sampler_key_init contract ---- */

static void test_key_init_contract(void) {
    mlx_stream s = mlxbridge_gpu_stream();

    /* seed < 0: unseeded, key_next yields empty subkey */
    sampler_key_t k;
    assert(sampler_key_init(&k, -1) == 0);
    assert(!k.seeded);
    mlx_array subkey = mlx_array_new();
    assert(sampler_key_next(&k, &subkey, s) == 0);
    sampler_key_free(&k);
    mlx_array_free(subkey);

    /* sampler_key_free on a zeroed struct is safe (double-free guard) */
    sampler_key_t k2;
    memset(&k2, 0, sizeof(k2));
    k2.key = mlx_array_new();
    k2.seeded = false;
    sampler_key_free(&k2);

    mlx_stream_free(s);
}

/* ---- Cycle 7: combined filters - all draws within the intersection set ---- */

static void test_combined_filters(void) {
    mlx_stream s = mlxbridge_gpu_stream();
    mlx_array logits = logits_1xv(kLogits);

    /* kLogits = {0.1, -2.0, 3.5, 1.0, 0.0, 2.9, -1.0, 0.5}
     * temperature = 2: scaled = {0.05, -1.0, 1.75, 0.5, 0.0, 1.45, -0.5, 0.25}
     * top_k = 4: keeps top-4 scaled values: idx 2 (1.75), 5 (1.45), 3 (0.5), 7 (0.25)
     * top_p = 0.9: softmax of kept = {0.44, 0.33, 0.13, 0.10}; cumsum from top:
     *   0.44, 0.77, 0.90, 1.0 - nucleus includes first 3 (idx 2, 5, 3)
     * min_p = 0.1: threshold on temp-scaled = max(1.75) + log(0.1) = 1.75 - 2.30 = -0.55
     *   All 3 survivors (1.75, 1.45, 0.5) > -0.55, so min_p is not further restrictive.
     * Final valid set: {2, 5, 3} */
    sampling_params_t sp = SAMPLING_PARAMS_DEFAULT;
    sp.temperature = 2.0f;
    sp.top_k = 4;
    sp.top_p = 0.9f;
    sp.min_p = 0.1f;
    sp.seed = 77;

    sampler_key_t key;
    assert(sampler_key_init(&key, sp.seed) == 0);

    for (int i = 0; i < 32; i++) {
        mlx_array subkey = mlx_array_new();
        assert(sampler_key_next(&key, &subkey, s) == 0);
        mlx_array tok = mlx_array_new();
        assert(sampler_sample_lazy(logits, &sp, subkey, false, s, &tok, NULL) == 0);
        int32_t id;
        assert(mlxbridge_item_int32(&id, tok) == 0);
        assert(id == 2 || id == 5 || id == 3);
        mlx_array_free(tok);
        mlx_array_free(subkey);
    }

    sampler_key_free(&key);

    /* Second block: min_p = 0.35 actually restricts.
     * temp=2: scaled = {0.05, -1.0, 1.75, 0.5, 0.0, 1.45, -0.5, 0.25}
     * min_p threshold on temp-scaled: max(1.75) + log(0.35) ~= 0.70
     * idx 3 (0.5) < 0.70 -> killed.  Survivors: {2 (1.75), 5 (1.45)} only. */
    {
        sampling_params_t sp2 = SAMPLING_PARAMS_DEFAULT;
        sp2.temperature = 2.0f;
        sp2.top_k = 4;
        sp2.top_p = 0.9f;
        sp2.min_p = 0.35f;
        sp2.seed = 78;

        sampler_key_t key2;
        assert(sampler_key_init(&key2, sp2.seed) == 0);

        for (int i = 0; i < 32; i++) {
            mlx_array subkey = mlx_array_new();
            assert(sampler_key_next(&key2, &subkey, s) == 0);
            mlx_array tok = mlx_array_new();
            assert(sampler_sample_lazy(logits, &sp2, subkey, false, s, &tok, NULL) == 0);
            int32_t id;
            assert(mlxbridge_item_int32(&id, tok) == 0);
            assert(id == 2 || id == 5);
            mlx_array_free(tok);
            mlx_array_free(subkey);
        }

        sampler_key_free(&key2);
    }

    mlx_array_free(logits);
    mlx_stream_free(s);
}

int main(void) {
    test_greedy_limit();
    test_top_k_filter();
    test_top_k1_equals_greedy();
    test_top_p_filter();
    test_top_p_zero_greedy();
    test_min_p_filter();
    test_logprob_computation();
    test_seeded_reproducibility();
    test_key_init_contract();
    test_combined_filters();
    printf("test_sampler_gpu: all tests passed\n");
    return 0;
}
