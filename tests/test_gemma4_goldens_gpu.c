#include "engine/forward.h"
#include "mlxbridge/mlxbridge.h"
#include "model/model.h"
#include "model/weights.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static mlx_stream gpu;

/* --- GELU approx golden (mlx.nn.gelu_approx) --- */
#define GELU_GOLDEN_N 16
static const float gelu_input[] = {
    -6.00000000e+00f, -5.18750000e+00f, -4.40625000e+00f, -3.59375000e+00f, -2.79687500e+00f,
    -2.00000000e+00f, -1.20312500e+00f, -4.00390625e-01f, 4.00390625e-01f, 1.20312500e+00f,
    2.00000000e+00f, 2.79687500e+00f, 3.59375000e+00f, 4.40625000e+00f, 5.18750000e+00f,
    6.00000000e+00f
};
static const float gelu_expected[] = {
    -0.00000000e+00f, -0.00000000e+00f, -0.00000000e+00f, -0.00000000e+00f, -5.46264648e-03f,
    -4.68750000e-02f, -1.38671875e-01f, -1.37695312e-01f, 2.63671875e-01f, 1.06250000e+00f,
    1.95312500e+00f, 2.79687500e+00f, 3.59375000e+00f, 4.40625000e+00f, 5.18750000e+00f,
    6.00000000e+00f
};

/* --- Proportional RoPE golden: fixture case --- */
#define ROPE_FIXTURE_H 2
#define ROPE_FIXTURE_S 3
#define ROPE_FIXTURE_HD 16
#define ROPE_FIXTURE_DIMS 16
#define ROPE_FIXTURE_OFFSET 2
#define ROPE_FIXTURE_N 96
#define ROPE_FIXTURE_NFREQS 8
static const float rope_fixture_freqs[] = {
    8.00000000e+00f, 4.49873047e+01f, 2.52982208e+02f, 1.42262354e+03f, INFINITY, INFINITY,
    INFINITY, INFINITY
};
static const float rope_fixture_input[] = {
    3.04687500e-01f, -1.03906250e+00f, 7.50000000e-01f, 9.41406250e-01f, -1.95312500e+00f,
    -1.30468750e+00f, 1.27929688e-01f, -3.16406250e-01f, -1.68457031e-02f, -8.51562500e-01f,
    8.78906250e-01f, 7.77343750e-01f, 6.59179688e-02f, 1.12500000e+00f, 4.66796875e-01f,
    -8.59375000e-01f, 3.69140625e-01f, -9.57031250e-01f, 8.78906250e-01f, -4.98046875e-02f,
    -1.84570312e-01f, -6.79687500e-01f, 1.21875000e+00f, -1.54296875e-01f, -4.27734375e-01f,
    -3.51562500e-01f, 5.31250000e-01f, 3.65234375e-01f, 4.12109375e-01f, 4.31640625e-01f,
    2.14062500e+00f, -4.06250000e-01f, -5.11718750e-01f, -8.12500000e-01f, 6.17187500e-01f,
    1.13281250e+00f, -1.13769531e-01f, -8.39843750e-01f, -8.24218750e-01f, 6.52343750e-01f,
    7.42187500e-01f, 5.42968750e-01f, -6.64062500e-01f, 2.32421875e-01f, 1.16699219e-01f,
    2.18750000e-01f, 8.71093750e-01f, 2.23632812e-01f, 6.79687500e-01f, 6.73828125e-02f,
    2.89062500e-01f, 6.32812500e-01f, -1.46093750e+00f, -3.20312500e-01f, -4.70703125e-01f,
    -6.40625000e-01f, -2.75390625e-01f, 1.49218750e+00f, -8.67187500e-01f, 9.68750000e-01f,
    -1.67968750e+00f, -3.33984375e-01f, 1.63085938e-01f, 5.85937500e-01f, 7.10937500e-01f,
    7.92968750e-01f, -3.49609375e-01f, -4.62890625e-01f, 8.59375000e-01f, -1.91406250e-01f,
    -1.27343750e+00f, -1.13281250e+00f, -9.17968750e-01f, 4.98046875e-01f, 1.42578125e-01f,
    6.91406250e-01f, -4.27734375e-01f, 1.58203125e-01f, 6.25000000e-01f, -3.08593750e-01f,
    4.57031250e-01f, -6.60156250e-01f, -3.63281250e-01f, -3.80859375e-01f, -1.19531250e+00f,
    4.86328125e-01f, -4.68750000e-01f, 1.25122070e-02f, 4.80468750e-01f, 4.47265625e-01f,
    6.64062500e-01f, -9.86328125e-02f, -4.23828125e-01f, -7.95898438e-02f, -1.68750000e+00f,
    -1.44531250e+00f
};
static const float rope_fixture_expected[] = {
    2.98828125e-01f, -1.00000000e+00f, 7.42187500e-01f, 9.41406250e-01f, -1.95312500e+00f,
    -1.30468750e+00f, 1.27929688e-01f, -3.16406250e-01f, 5.90820312e-02f, -8.98437500e-01f,
    8.86718750e-01f, 7.77343750e-01f, 6.59179688e-02f, 1.12500000e+00f, 4.66796875e-01f,
    -8.59375000e-01f, 5.00000000e-01f, -9.29687500e-01f, 8.71093750e-01f, -5.05371094e-02f,
    -1.84570312e-01f, -6.79687500e-01f, 1.21875000e+00f, -1.54296875e-01f, -2.63671875e-01f,
    -4.14062500e-01f, 5.42968750e-01f, 3.65234375e-01f, 4.12109375e-01f, 4.31640625e-01f,
    2.14062500e+00f, -4.06250000e-01f, -8.04687500e-01f, -8.59375000e-01f, 6.28906250e-01f,
    1.13281250e+00f, -1.13769531e-01f, -8.39843750e-01f, -8.24218750e-01f, 6.52343750e-01f,
    4.06250000e-01f, 4.68750000e-01f, -6.52343750e-01f, 2.35351562e-01f, 1.16699219e-01f,
    2.18750000e-01f, 8.71093750e-01f, 2.23632812e-01f, 7.26562500e-01f, 9.99450684e-04f,
    2.96875000e-01f, 6.32812500e-01f, -1.46093750e+00f, -3.20312500e-01f, -4.70703125e-01f,
    -6.40625000e-01f, -9.86328125e-02f, 1.49218750e+00f, -8.63281250e-01f, 9.68750000e-01f,
    -1.67968750e+00f, -3.33984375e-01f, 1.63085938e-01f, 5.85937500e-01f, 9.96093750e-01f,
    7.57812500e-01f, -3.51562500e-01f, -4.64843750e-01f, 8.59375000e-01f, -1.91406250e-01f,
    -1.27343750e+00f, -1.13281250e+00f, -5.93750000e-01f, 5.50781250e-01f, 1.38671875e-01f,
    6.91406250e-01f, -4.27734375e-01f, 1.58203125e-01f, 6.25000000e-01f, -3.08593750e-01f,
    1.70898438e-01f, -6.95312500e-01f, -3.73046875e-01f, -3.80859375e-01f, -1.19531250e+00f,
    4.86328125e-01f, -4.68750000e-01f, 1.25122070e-02f, 6.40625000e-01f, 3.86718750e-01f,
    6.60156250e-01f, -9.96093750e-02f, -4.23828125e-01f, -7.95898438e-02f, -1.68750000e+00f,
    -1.44531250e+00f
};

/* --- Proportional RoPE golden: e2b case --- */
#define ROPE_E2B_H 2
#define ROPE_E2B_S 3
#define ROPE_E2B_HD 16
#define ROPE_E2B_DIMS 16
#define ROPE_E2B_OFFSET 2
#define ROPE_E2B_N 96
#define ROPE_E2B_NFREQS 8
static const float rope_e2b_freqs[] = {
    1.00000000e+00f, 5.62341309e+00f, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY
};
static const float rope_e2b_input[] = {
    3.04687500e-01f, -1.03906250e+00f, 7.50000000e-01f, 9.41406250e-01f, -1.95312500e+00f,
    -1.30468750e+00f, 1.27929688e-01f, -3.16406250e-01f, -1.68457031e-02f, -8.51562500e-01f,
    8.78906250e-01f, 7.77343750e-01f, 6.59179688e-02f, 1.12500000e+00f, 4.66796875e-01f,
    -8.59375000e-01f, 3.69140625e-01f, -9.57031250e-01f, 8.78906250e-01f, -4.98046875e-02f,
    -1.84570312e-01f, -6.79687500e-01f, 1.21875000e+00f, -1.54296875e-01f, -4.27734375e-01f,
    -3.51562500e-01f, 5.31250000e-01f, 3.65234375e-01f, 4.12109375e-01f, 4.31640625e-01f,
    2.14062500e+00f, -4.06250000e-01f, -5.11718750e-01f, -8.12500000e-01f, 6.17187500e-01f,
    1.13281250e+00f, -1.13769531e-01f, -8.39843750e-01f, -8.24218750e-01f, 6.52343750e-01f,
    7.42187500e-01f, 5.42968750e-01f, -6.64062500e-01f, 2.32421875e-01f, 1.16699219e-01f,
    2.18750000e-01f, 8.71093750e-01f, 2.23632812e-01f, 6.79687500e-01f, 6.73828125e-02f,
    2.89062500e-01f, 6.32812500e-01f, -1.46093750e+00f, -3.20312500e-01f, -4.70703125e-01f,
    -6.40625000e-01f, -2.75390625e-01f, 1.49218750e+00f, -8.67187500e-01f, 9.68750000e-01f,
    -1.67968750e+00f, -3.33984375e-01f, 1.63085938e-01f, 5.85937500e-01f, 7.10937500e-01f,
    7.92968750e-01f, -3.49609375e-01f, -4.62890625e-01f, 8.59375000e-01f, -1.91406250e-01f,
    -1.27343750e+00f, -1.13281250e+00f, -9.17968750e-01f, 4.98046875e-01f, 1.42578125e-01f,
    6.91406250e-01f, -4.27734375e-01f, 1.58203125e-01f, 6.25000000e-01f, -3.08593750e-01f,
    4.57031250e-01f, -6.60156250e-01f, -3.63281250e-01f, -3.80859375e-01f, -1.19531250e+00f,
    4.86328125e-01f, -4.68750000e-01f, 1.25122070e-02f, 4.80468750e-01f, 4.47265625e-01f,
    6.64062500e-01f, -9.86328125e-02f, -4.23828125e-01f, -7.95898438e-02f, -1.68750000e+00f,
    -1.44531250e+00f
};
static const float rope_e2b_expected[] = {
    -1.11328125e-01f, -6.75781250e-01f, 7.50000000e-01f, 9.41406250e-01f, -1.95312500e+00f,
    -1.30468750e+00f, 1.27929688e-01f, -3.16406250e-01f, 2.83203125e-01f, -1.15625000e+00f,
    8.78906250e-01f, 7.77343750e-01f, 6.59179688e-02f, 1.12500000e+00f, 4.66796875e-01f,
    -8.59375000e-01f, -3.04687500e-01f, -6.44531250e-01f, 8.78906250e-01f, -4.98046875e-02f,
    -1.84570312e-01f, -6.79687500e-01f, 1.21875000e+00f, -1.54296875e-01f, 4.74609375e-01f,
    -7.89062500e-01f, 5.31250000e-01f, 3.65234375e-01f, 4.12109375e-01f, 4.31640625e-01f,
    2.14062500e+00f, -4.06250000e-01f, 8.94531250e-01f, -9.68750000e-01f, 6.17187500e-01f,
    1.13281250e+00f, -1.13769531e-01f, -8.39843750e-01f, -8.24218750e-01f, 6.52343750e-01f,
    -9.76562500e-02f, -1.19140625e-01f, -6.64062500e-01f, 2.32421875e-01f, 1.16699219e-01f,
    2.18750000e-01f, 8.71093750e-01f, 2.23632812e-01f, -3.24707031e-02f, -4.57031250e-01f,
    2.89062500e-01f, 6.32812500e-01f, -1.46093750e+00f, -3.20312500e-01f, -4.70703125e-01f,
    -6.40625000e-01f, 7.34375000e-01f, 1.42187500e+00f, -8.67187500e-01f, 9.68750000e-01f,
    -1.67968750e+00f, -3.33984375e-01f, 1.63085938e-01f, 5.85937500e-01f, -5.74218750e-01f,
    4.29687500e-01f, -3.49609375e-01f, -4.62890625e-01f, 8.59375000e-01f, -1.91406250e-01f,
    -1.27343750e+00f, -1.13281250e+00f, 1.00781250e+00f, 8.32031250e-01f, 1.42578125e-01f,
    6.91406250e-01f, -4.27734375e-01f, 1.58203125e-01f, 6.25000000e-01f, -3.08593750e-01f,
    6.49414062e-02f, -7.92968750e-01f, -3.63281250e-01f, -3.80859375e-01f, -1.19531250e+00f,
    4.86328125e-01f, -4.68750000e-01f, 1.25122070e-02f, -6.60156250e-01f, -9.22851562e-02f,
    6.64062500e-01f, -9.86328125e-02f, -4.23828125e-01f, -7.95898438e-02f, -1.68750000e+00f,
    -1.44531250e+00f
};

/* --- PLE inputs golden (mlx-lm _get/_project_per_layer_inputs) --- */
#define PLE_INPUTS_B 1
#define PLE_INPUTS_S 2
#define PLE_INPUTS_HIDDEN 8
#define PLE_INPUTS_PLE_DIM 4
#define PLE_INPUTS_LAYERS 2
#define PLE_INPUTS_VOCAB 3
#define PLE_INPUTS_N 16
static const float ple_inputs_h[] = {
    -1.10156250e+00f, 4.27734375e-01f, -2.20703125e-01f, 1.13769531e-01f, 6.29882812e-02f,
    1.19531250e+00f, 5.54687500e-01f, -8.67187500e-01f, 8.00781250e-02f, -1.75781250e-02f,
    -7.96875000e-01f, 2.04101562e-01f, -2.28271484e-02f, 1.35937500e+00f, -1.87500000e-01f,
    4.41406250e-01f
};
static const float ple_inputs_emb_w[] = {
    8.25195312e-02f, -4.64843750e-01f, 5.05371094e-02f, 6.87500000e-01f, -1.75781250e+00f,
    1.68750000e+00f, -4.57031250e-01f, -5.97656250e-01f, -1.04687500e+00f, 9.33593750e-01f,
    6.75781250e-01f, 1.24218750e+00f, 8.94531250e-01f, 2.63671875e-01f, 3.28125000e-01f,
    9.33593750e-01f, -8.78906250e-01f, -4.58984375e-02f, 3.82812500e-01f, -4.53125000e-01f,
    7.22656250e-01f, -3.51562500e-01f, 6.71875000e-01f, 1.40625000e-01f
};
static const float ple_inputs_proj_w[] = {
    4.62890625e-01f, -1.51562500e+00f, -8.59375000e-01f, 1.34375000e+00f, 1.77734375e-01f,
    -8.15429688e-02f, 9.64843750e-01f, 7.50000000e-01f, -4.68750000e-02f, -6.44531250e-01f,
    1.96093750e+00f, 6.91406250e-01f, -1.57031250e+00f, 8.39843750e-01f, 7.69531250e-01f,
    8.12500000e-01f, -4.04296875e-01f, 1.46875000e+00f, -7.46093750e-01f, 1.21093750e+00f,
    2.92968750e-01f, 1.69531250e+00f, -3.88671875e-01f, 6.95312500e-01f, 8.43750000e-01f,
    -3.24218750e-01f, 1.12304688e-02f, -4.14062500e-01f, 4.78515625e-01f, 6.87500000e-01f,
    -2.91015625e-01f, 3.45703125e-01f, -5.82031250e-01f, -5.19531250e-01f, -1.92187500e+00f,
    -1.17187500e+00f, -6.75781250e-01f, 1.08398438e-01f, 1.52343750e+00f, 2.69531250e-01f,
    9.13085938e-02f, 3.47656250e-01f, -1.39843750e+00f, 4.83398438e-02f, -8.67187500e-01f,
    -5.74218750e-01f, 2.03125000e+00f, 1.84375000e+00f, 9.76562500e-01f, 5.42968750e-01f,
    5.03906250e-01f, -9.64843750e-01f, -1.25781250e+00f, 3.33984375e-01f, -4.47265625e-01f,
    -7.77343750e-01f, -8.05664062e-02f, -7.22656250e-02f, 1.89453125e-01f, -7.38281250e-01f,
    1.73339844e-02f, -1.24218750e+00f, 1.22656250e+00f, 1.51562500e+00f
};
static const float ple_inputs_expected[] = {
    -3.80859375e-01f, -6.32812500e-01f, 1.35156250e+00f, 6.17187500e-01f, -1.76562500e+00f,
    1.94531250e+00f, -8.04687500e-01f, -1.96875000e+00f, -1.08593750e+00f, 1.31250000e+00f,
    2.25000000e+00f, 2.17187500e+00f, 2.07812500e+00f, 8.94531250e-01f, 2.63671875e-01f,
    3.04687500e-01f
};

/* --- PLE apply golden (per-layer gate/project/norm/residual) --- */
#define PLE_APPLY_B 1
#define PLE_APPLY_S 2
#define PLE_APPLY_HIDDEN 8
#define PLE_APPLY_PLE_DIM 4
#define PLE_APPLY_LAYERS 2
#define PLE_APPLY_LAYER 0
#define PLE_APPLY_N 16
#define PLE_APPLY_PLE_N 16
static const float ple_apply_h[] = {
    -1.10351562e-01f, 1.64062500e+00f, -3.39843750e-01f, -1.21093750e+00f, -9.08203125e-02f,
    8.55468750e-01f, 1.93359375e-01f, -2.67028809e-03f, -1.40625000e+00f, 5.00000000e-01f,
    4.86328125e-01f, 1.34375000e+00f, 1.49218750e+00f, 6.87500000e-01f, -1.64062500e+00f,
    -1.43750000e+00f
};
static const float ple_apply_ple_inputs[] = {
    -7.89062500e-01f, -2.03125000e+00f, 6.01562500e-01f, 7.46093750e-01f, -3.10546875e-01f,
    3.67187500e-01f, 1.71093750e+00f, 1.06250000e+00f, 7.07031250e-01f, 6.87500000e-01f,
    -8.63281250e-01f, 9.64843750e-01f, -1.64843750e+00f, -3.32031250e-01f, -4.37500000e-01f,
    -1.72656250e+00f
};
static const float ple_apply_gate_w[] = {
    -1.10156250e+00f, 2.94921875e-01f, -7.89062500e-01f, 1.31250000e+00f, 2.44140625e-01f,
    3.65234375e-01f, 7.50000000e-01f, 3.39843750e-01f, -1.41406250e+00f, 2.45312500e+00f,
    1.46093750e+00f, 1.52343750e+00f, -7.46093750e-01f, -1.84375000e+00f, -1.50781250e+00f,
    -1.05468750e+00f, 8.82812500e-01f, 8.98437500e-01f, 1.46093750e+00f, 2.14062500e+00f,
    5.19531250e-01f, 3.55468750e-01f, -3.63281250e-01f, -3.35937500e-01f, -1.15625000e+00f,
    5.11718750e-01f, 3.18359375e-01f, -4.04296875e-01f, -8.12500000e-01f, 4.57031250e-01f,
    1.30468750e+00f, 8.15429688e-02f
};
static const float ple_apply_proj_w[] = {
    2.83203125e-01f, -5.46875000e-01f, -2.48046875e-01f, -3.16406250e-01f, -2.85156250e-01f,
    -5.97656250e-01f, 3.96484375e-01f, 1.29687500e+00f, -9.45312500e-01f, 1.39843750e+00f,
    -4.02343750e-01f, -3.67187500e-01f, -2.15625000e+00f, -2.42187500e+00f, 7.34375000e-01f,
    -1.09375000e+00f, 8.04687500e-01f, 5.05371094e-02f, 2.50000000e-01f, 1.33593750e+00f,
    -7.22656250e-02f, -3.26562500e+00f, -1.74218750e+00f, 9.72656250e-01f, 2.00000000e+00f,
    -9.61914062e-02f, -7.03125000e-01f, -1.39062500e+00f, 5.46875000e-01f, -1.79687500e+00f,
    -1.24218750e+00f, 1.35742188e-01f
};
static const float ple_apply_expected[] = {
    -3.82812500e-01f, 2.90625000e+00f, -7.81250000e-01f, -2.37500000e+00f, 1.27343750e+00f,
    1.98437500e+00f, -1.03906250e+00f, 2.65625000e-01f, -1.57031250e+00f, -1.25000000e-01f,
    1.39843750e+00f, -8.59375000e-01f, 1.51562500e+00f, -4.68750000e-01f, -1.00781250e+00f,
    -1.83593750e+00f
};

/* --- Softcap golden (tanh(x/cap)*cap) --- */
#define SOFTCAP_GOLDEN_N 16
static const float softcap_input[] = {
    -6.00000000e+01f, -5.20000000e+01f, -4.40000000e+01f, -3.60000000e+01f, -2.80000000e+01f,
    -2.00000000e+01f, -1.20000000e+01f, -4.00000000e+00f, 4.00000000e+00f, 1.20000000e+01f,
    2.00000000e+01f, 2.80000000e+01f, 3.60000000e+01f, 4.40000000e+01f, 5.20000000e+01f,
    6.00000000e+01f
};
static const float softcap_expected[] = {
    -2.90000000e+01f, -2.82500000e+01f, -2.70000000e+01f, -2.51250000e+01f, -2.18750000e+01f,
    -1.75000000e+01f, -1.14375000e+01f, -3.98437500e+00f, 3.98437500e+00f, 1.14375000e+01f,
    1.75000000e+01f, 2.18750000e+01f, 2.51250000e+01f, 2.70000000e+01f, 2.82500000e+01f,
    2.90000000e+01f
};

/* ---- GELU approx test ---- */

static void test_gelu_approx_golden(void) {
    int shape[] = {1, GELU_GOLDEN_N};
    mlx_array x = mlx_array_new_data(gelu_input, shape, 2, MLX_FLOAT32);
    mlx_array xbf = mlx_array_new();
    MLXB_CHECK(mlx_astype(&xbf, x, MLX_BFLOAT16, gpu));

    mlx_array out = mlx_array_new();
    assert(fwd_gelu_approx(&out, xbf, gpu) == 0);

    mlx_array outf = mlx_array_new();
    MLXB_CHECK(mlx_astype(&outf, out, MLX_FLOAT32, gpu));
    assert(MLXB_CHECK(mlx_array_eval(outf)));

    const float *d = mlx_array_data_float32(outf);
    for (int i = 0; i < GELU_GOLDEN_N; i++) {
        float diff = fabsf(d[i] - gelu_expected[i]);
        if (diff > 5e-3f) {
            fprintf(stderr, "gelu[%d]: got %f, expected %f (diff %f)\n",
                    i, (double)d[i], (double)gelu_expected[i], (double)diff);
        }
        assert(diff <= 5e-3f);
    }

    mlx_array_free(outf);
    mlx_array_free(out);
    mlx_array_free(xbf);
    mlx_array_free(x);
}

/* ---- Proportional RoPE test helper ---- */

static void test_rope_case(const float *freqs_data, int n_freqs,
                           const float *input_data, const float *expected_data,
                           int H, int S, int head_dim, int dims, int offset,
                           float rope_theta, float partial_rotary,
                           float prop_factor, const char *label) {
    int shape[] = {1, H, S, head_dim};
    mlx_array x = mlx_array_new_data(input_data, shape, 4, MLX_FLOAT32);
    mlx_array xbf = mlx_array_new();
    MLXB_CHECK(mlx_astype(&xbf, x, MLX_BFLOAT16, gpu));

    model_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.head_dim = head_dim;
    cfg.global_head_dim = head_dim;
    cfg.rope_theta = rope_theta;
    cfg.rope_proportional = true;
    cfg.rope_proportional_factor = prop_factor;
    cfg.partial_rotary_factor_global = partial_rotary;
    cfg.num_hidden_layers = 1;
    cfg.layer_is_global[0] = true;
    cfg.has_explicit_layer_types = true;

    /* Verify fwd_rope_proportional_freqs matches the oracle freqs */
    float computed_freqs[16];
    assert(n_freqs == head_dim / 2);
    assert(fwd_rope_proportional_freqs(&cfg, computed_freqs, n_freqs) == 0);
    for (int i = 0; i < n_freqs; i++) {
        if (isinf(freqs_data[i])) {
            assert(isinf(computed_freqs[i]));
        } else {
            float diff = fabsf(computed_freqs[i] - freqs_data[i]);
            if (diff > 1e-3f) {
                fprintf(stderr, "rope %s freqs[%d]: C=%f oracle=%f (diff %f)\n",
                        label, i, (double)computed_freqs[i],
                        (double)freqs_data[i], (double)diff);
            }
            assert(diff <= 1e-3f);
        }
    }

    /* Build freqs array and apply rope */
    int fshape[] = {n_freqs};
    mlx_array freqs = mlx_array_new_data(freqs_data, fshape, 1, MLX_FLOAT32);

    mlx_array out = mlx_array_new();
    assert(fwd_rope_apply(&out, xbf, dims, &cfg, 0, offset, freqs, gpu) == 0);

    mlx_array outf = mlx_array_new();
    MLXB_CHECK(mlx_astype(&outf, out, MLX_FLOAT32, gpu));
    assert(MLXB_CHECK(mlx_array_eval(outf)));

    const float *d = mlx_array_data_float32(outf);
    int n = 1 * H * S * head_dim;
    float max_diff = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = fabsf(d[i] - expected_data[i]);
        if (diff > max_diff) max_diff = diff;
    }
    if (max_diff > 5e-3f) {
        fprintf(stderr, "rope %s: max_diff = %f\n", label, (double)max_diff);
    }
    assert(max_diff <= 5e-3f);

    mlx_array_free(outf);
    mlx_array_free(out);
    mlx_array_free(freqs);
    mlx_array_free(xbf);
    mlx_array_free(x);
}

static void test_softcap_golden(void) {
    int shape[] = {1, SOFTCAP_GOLDEN_N};
    mlx_array x = mlx_array_new_data(softcap_input, shape, 2, MLX_FLOAT32);
    mlx_array xbf = mlx_array_new();
    MLXB_CHECK(mlx_astype(&xbf, x, MLX_BFLOAT16, gpu));

    mlx_array out = mlx_array_new();
    assert(fwd_softcap(&out, xbf, 30.0f, gpu) == 0);

    mlx_array outf = mlx_array_new();
    MLXB_CHECK(mlx_astype(&outf, out, MLX_FLOAT32, gpu));
    assert(MLXB_CHECK(mlx_array_eval(outf)));

    const float *d = mlx_array_data_float32(outf);
    for (int i = 0; i < SOFTCAP_GOLDEN_N; i++) {
        float diff = fabsf(d[i] - softcap_expected[i]);
        if (diff > 5e-2f) {
            fprintf(stderr, "softcap[%d]: got %f, expected %f (diff %f)\n",
                    i, (double)d[i], (double)softcap_expected[i], (double)diff);
        }
        assert(diff <= 5e-2f);
    }

    mlx_array_free(outf);
    mlx_array_free(out);
    mlx_array_free(xbf);
    mlx_array_free(x);
}

static void test_rope_proportional_fixture(void) {
    test_rope_case(rope_fixture_freqs, ROPE_FIXTURE_NFREQS,
                   rope_fixture_input, rope_fixture_expected,
                   ROPE_FIXTURE_H, ROPE_FIXTURE_S, ROPE_FIXTURE_HD,
                   ROPE_FIXTURE_DIMS, ROPE_FIXTURE_OFFSET,
                   1000000.0f, 0.5f, 8.0f, "fixture");
}

static void test_rope_proportional_e2b(void) {
    test_rope_case(rope_e2b_freqs, ROPE_E2B_NFREQS,
                   rope_e2b_input, rope_e2b_expected,
                   ROPE_E2B_H, ROPE_E2B_S, ROPE_E2B_HD,
                   ROPE_E2B_DIMS, ROPE_E2B_OFFSET,
                   1000000.0f, 0.25f, 1.0f, "e2b");
}

/* Helper: insert a bf16 tensor from f32 data into a weight map */
static void insert_bf16(mlx_map_string_to_array m, const char *name,
                        const float *data, const int *shape, int ndim) {
    mlx_array f32 = mlx_array_new_data(data, shape, ndim, MLX_FLOAT32);
    mlx_array bf = mlx_array_new();
    MLXB_CHECK(mlx_astype(&bf, f32, MLX_BFLOAT16, gpu));
    mlx_array_free(f32);
    MLXB_CHECK(mlx_map_string_to_array_insert(m, name, bf));
    mlx_array_free(bf);
}

static void test_ple_inputs_golden(void) {
    model_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.family = MODEL_GEMMA4;
    cfg.weight_prefix = "model";
    cfg.hidden_size = PLE_INPUTS_HIDDEN;
    cfg.hidden_size_per_layer_input = PLE_INPUTS_PLE_DIM;
    cfg.num_hidden_layers = PLE_INPUTS_LAYERS;
    cfg.rms_norm_eps = 1e-6f;
    cfg.norm_has_offset = false;

    mlx_map_string_to_array m = mlx_map_string_to_array_new();
    int emb_shape[] = {PLE_INPUTS_VOCAB, PLE_INPUTS_LAYERS * PLE_INPUTS_PLE_DIM};
    insert_bf16(m, "model.embed_tokens_per_layer.weight",
                ple_inputs_emb_w, emb_shape, 2);
    int proj_shape[] = {PLE_INPUTS_LAYERS * PLE_INPUTS_PLE_DIM, PLE_INPUTS_HIDDEN};
    insert_bf16(m, "model.per_layer_model_projection.weight",
                ple_inputs_proj_w, proj_shape, 2);
    int norm_shape[] = {PLE_INPUTS_PLE_DIM};
    float ones[PLE_INPUTS_PLE_DIM];
    for (int i = 0; i < PLE_INPUTS_PLE_DIM; i++) ones[i] = 1.0f;
    insert_bf16(m, "model.per_layer_projection_norm.weight",
                ones, norm_shape, 1);

    weights_t w = { .params = m, .count = 3, .total_bytes = 0 };

    int ids_data[] = {0, 1};
    int ids_shape[] = {PLE_INPUTS_B, PLE_INPUTS_S};
    mlx_array ids = mlx_array_new_data(ids_data, ids_shape, 2, MLX_INT32);

    int h_shape[] = {PLE_INPUTS_B, PLE_INPUTS_S, PLE_INPUTS_HIDDEN};
    mlx_array h_f32 = mlx_array_new_data(ple_inputs_h, h_shape, 3, MLX_FLOAT32);
    mlx_array h = mlx_array_new();
    MLXB_CHECK(mlx_astype(&h, h_f32, MLX_BFLOAT16, gpu));

    mlx_array out = mlx_array_new();
    assert(fwd_ple_inputs(&out, ids, h, &w, &cfg, gpu) == 0);

    mlx_array outf = mlx_array_new();
    MLXB_CHECK(mlx_astype(&outf, out, MLX_FLOAT32, gpu));
    assert(MLXB_CHECK(mlx_array_eval(outf)));

    const float *d = mlx_array_data_float32(outf);
    float max_diff = 0.0f;
    for (int i = 0; i < PLE_INPUTS_N; i++) {
        float diff = fabsf(d[i] - ple_inputs_expected[i]);
        if (diff > max_diff) max_diff = diff;
    }
    if (max_diff > 5e-2f) {
        fprintf(stderr, "ple_inputs: max_diff = %f\n", (double)max_diff);
        for (int i = 0; i < PLE_INPUTS_N; i++)
            fprintf(stderr, "  [%d] got=%f exp=%f\n", i, (double)d[i],
                    (double)ple_inputs_expected[i]);
    }
    assert(max_diff <= 5e-2f);

    mlx_array_free(outf);
    mlx_array_free(out);
    mlx_array_free(h);
    mlx_array_free(h_f32);
    mlx_array_free(ids);
    mlx_map_string_to_array_free(m);
}

static void test_ple_apply_golden(void) {
    model_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.family = MODEL_GEMMA4;
    cfg.weight_prefix = "model";
    cfg.hidden_size = PLE_APPLY_HIDDEN;
    cfg.hidden_size_per_layer_input = PLE_APPLY_PLE_DIM;
    cfg.num_hidden_layers = PLE_APPLY_LAYERS;
    cfg.rms_norm_eps = 1e-6f;
    cfg.norm_has_offset = false;

    mlx_map_string_to_array m = mlx_map_string_to_array_new();
    char name[256];

    int gate_shape[] = {PLE_APPLY_PLE_DIM, PLE_APPLY_HIDDEN};
    snprintf(name, sizeof(name), "model.layers.%d.per_layer_input_gate.weight",
             PLE_APPLY_LAYER);
    insert_bf16(m, name, ple_apply_gate_w, gate_shape, 2);

    int proj_shape[] = {PLE_APPLY_HIDDEN, PLE_APPLY_PLE_DIM};
    snprintf(name, sizeof(name), "model.layers.%d.per_layer_projection.weight",
             PLE_APPLY_LAYER);
    insert_bf16(m, name, ple_apply_proj_w, proj_shape, 2);

    int norm_shape[] = {PLE_APPLY_HIDDEN};
    float ones_arr[PLE_APPLY_HIDDEN];
    for (int i = 0; i < PLE_APPLY_HIDDEN; i++) ones_arr[i] = 1.0f;
    snprintf(name, sizeof(name), "model.layers.%d.post_per_layer_input_norm.weight",
             PLE_APPLY_LAYER);
    insert_bf16(m, name, ones_arr, norm_shape, 1);

    weights_t w = { .params = m, .count = 3, .total_bytes = 0 };

    int ple_shape[] = {PLE_APPLY_B, PLE_APPLY_S, PLE_APPLY_LAYERS, PLE_APPLY_PLE_DIM};
    mlx_array ple_f32 = mlx_array_new_data(ple_apply_ple_inputs, ple_shape, 4,
                                            MLX_FLOAT32);
    mlx_array ple = mlx_array_new();
    MLXB_CHECK(mlx_astype(&ple, ple_f32, MLX_BFLOAT16, gpu));

    int h_shape[] = {PLE_APPLY_B, PLE_APPLY_S, PLE_APPLY_HIDDEN};
    mlx_array h_f32 = mlx_array_new_data(ple_apply_h, h_shape, 3, MLX_FLOAT32);
    mlx_array h = mlx_array_new();
    MLXB_CHECK(mlx_astype(&h, h_f32, MLX_BFLOAT16, gpu));

    assert(fwd_ple_apply(&h, PLE_APPLY_LAYER, ple, &w, &cfg, gpu) == 0);

    mlx_array outf = mlx_array_new();
    MLXB_CHECK(mlx_astype(&outf, h, MLX_FLOAT32, gpu));
    assert(MLXB_CHECK(mlx_array_eval(outf)));

    const float *d = mlx_array_data_float32(outf);
    float max_diff = 0.0f;
    for (int i = 0; i < PLE_APPLY_N; i++) {
        float diff = fabsf(d[i] - ple_apply_expected[i]);
        if (diff > max_diff) max_diff = diff;
    }
    if (max_diff > 5e-2f) {
        fprintf(stderr, "ple_apply: max_diff = %f\n", (double)max_diff);
        for (int i = 0; i < PLE_APPLY_N; i++)
            fprintf(stderr, "  [%d] got=%f exp=%f\n", i, (double)d[i],
                    (double)ple_apply_expected[i]);
    }
    assert(max_diff <= 5e-2f);

    mlx_array_free(outf);
    mlx_array_free(h);
    mlx_array_free(h_f32);
    mlx_array_free(ple);
    mlx_array_free(ple_f32);
    mlx_map_string_to_array_free(m);
}

int main(void) {
    gpu = mlxbridge_gpu_stream();

    test_gelu_approx_golden();
    printf("  test_gelu_approx_golden: passed\n");

    test_softcap_golden();
    printf("  test_softcap_golden: passed\n");

    test_rope_proportional_fixture();
    printf("  test_rope_proportional_fixture: passed\n");

    test_rope_proportional_e2b();
    printf("  test_rope_proportional_e2b: passed\n");

    test_ple_inputs_golden();
    printf("  test_ple_inputs_golden: passed\n");

    test_ple_apply_golden();
    printf("  test_ple_apply_golden: passed\n");

    printf("test_gemma4_goldens_gpu: all passed\n");
    return 0;
}
