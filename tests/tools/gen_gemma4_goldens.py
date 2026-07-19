#!/usr/bin/env python3
"""Dev-time oracle for gemma4 GELU approx and proportional RoPE goldens.

Generates ground-truth C float array literals from mlx-lm / mlx primitives.
Oracle output is ground truth: a C-side mismatch is a C bug; vectors are
never edited to fit.

Run from the repo root:
    uv run --with mlx-lm python tests/tools/gen_gemma4_goldens.py
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np

try:
    from mlx_lm.models.rope_utils import ProportionalRoPE
except ImportError:
    raise SystemExit(
        "mlx-lm is required: pip install mlx-lm (or uv run --with mlx-lm)"
    )


def fmt_c_array(name: str, arr: np.ndarray) -> str:
    vals = arr.flatten()
    lines = [f"static const float {name}[] = {{"]
    row = "    "
    for i, v in enumerate(vals):
        entry = f"{v:.8e}f"
        if i < len(vals) - 1:
            entry += ", "
        if len(row) + len(entry) > 100:
            lines.append(row)
            row = "    "
        row += entry
    lines.append(row)
    lines.append("};")
    return "\n".join(lines)


def gen_gelu_approx():
    """Golden for fwd_gelu_approx: 16 values spanning [-6, 6]."""
    print("/* --- GELU approx golden (mlx.nn.gelu_approx) --- */")
    x_np = np.linspace(-6.0, 6.0, 16, dtype=np.float32)
    x_bf16 = mx.array(x_np).astype(mx.bfloat16)
    out = nn.gelu_approx(x_bf16)
    out_f32 = np.array(out.astype(mx.float32))
    in_f32 = np.array(x_bf16.astype(mx.float32))

    print(f"#define GELU_GOLDEN_N 16")
    print(fmt_c_array("gelu_input", in_f32))
    print(fmt_c_array("gelu_expected", out_f32))
    print()


def gen_proportional_rope():
    """Golden for fwd_rope_apply with proportional freqs.

    Uses the real mlx_lm.models.rope_utils.ProportionalRoPE as oracle.
    Two cases:
    1) fixture-shaped: partial_rotary=0.5, factor=8.0, head_dim=16
    2) E2B-shaped: partial_rotary=0.25, factor=1.0, head_dim=16
    """
    print("/* --- Proportional RoPE golden (mlx_lm ProportionalRoPE oracle) --- */")

    for case_name, partial_rotary, factor, base in [
        ("fixture", 0.5, 8.0, 1000000.0),
        ("e2b", 0.25, 1.0, 1000000.0),
    ]:
        head_dim = 16
        rotated_dims = int(partial_rotary * head_dim)
        n_freqs = head_dim // 2
        H = 2
        S = 3
        offset = 2

        rope = ProportionalRoPE(
            dims=head_dim,
            rotated_dims=rotated_dims,
            traditional=False,
            base=base,
            factor=factor,
        )

        freqs_np = np.array(rope._freqs)

        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((1, H, S, head_dim)).astype(np.float32)
        x_bf16 = mx.array(x_np).astype(mx.bfloat16)

        out = rope(x_bf16, offset=offset)
        out_f32 = np.array(out.astype(mx.float32))
        in_f32 = np.array(x_bf16.astype(mx.float32))

        tag = case_name
        print(f"/* Case: {case_name} (partial_rotary={partial_rotary}, factor={factor}) */")
        print(f"#define ROPE_{tag.upper()}_H {H}")
        print(f"#define ROPE_{tag.upper()}_S {S}")
        print(f"#define ROPE_{tag.upper()}_HD {head_dim}")
        print(f"#define ROPE_{tag.upper()}_DIMS {head_dim}")
        print(f"#define ROPE_{tag.upper()}_OFFSET {offset}")
        print(f"#define ROPE_{tag.upper()}_N {1 * H * S * head_dim}")
        print(f"#define ROPE_{tag.upper()}_NFREQS {n_freqs}")
        print(fmt_c_array(f"rope_{tag}_freqs", freqs_np))
        print(fmt_c_array(f"rope_{tag}_input", in_f32))
        print(fmt_c_array(f"rope_{tag}_expected", out_f32))
        print()


def gen_softcap():
    """Golden for fwd_softcap: bf16 input spanning +-60, cap=30."""
    print("/* --- Softcap golden (tanh(x/cap)*cap) --- */")
    x_np = np.linspace(-60.0, 60.0, 16, dtype=np.float32)
    x_bf16 = mx.array(x_np).astype(mx.bfloat16)
    cap = mx.array(30.0).astype(mx.bfloat16)
    out = mx.tanh(x_bf16 / cap) * cap
    out_f32 = np.array(out.astype(mx.float32))
    in_f32 = np.array(x_bf16.astype(mx.float32))

    print(f"#define SOFTCAP_GOLDEN_N 16")
    print(fmt_c_array("softcap_input", in_f32))
    print(fmt_c_array("softcap_expected", out_f32))
    print()


def gen_ple_inputs():
    """Golden for fwd_ple_inputs: reimplements _get_per_layer_inputs + _project_per_layer_inputs.

    Faithful reimplementation of the mlx-lm gemma4_text PLE math using raw mlx
    primitives - not a live call into mlx_lm modules. Uses small fixed tensors
    to lock the three scale factors:
      sqrt(ple_dim), 1/sqrt(hidden_size), 1/sqrt(2)
    and the RMSNorm + residual combination.
    """
    print("/* --- PLE inputs golden (mlx-lm _get/_project_per_layer_inputs) --- */")

    B, S = 1, 2
    hidden_size = 8
    ple_dim = 4
    num_layers = 2
    eps = 1e-6

    rng = np.random.default_rng(99)

    ids = np.array([[0, 1]], dtype=np.int32)
    emb_per_layer_w = rng.standard_normal((3, num_layers * ple_dim)).astype(np.float32)
    proj_w = rng.standard_normal((num_layers * ple_dim, hidden_size)).astype(np.float32)
    proj_norm_w = np.ones(ple_dim, dtype=np.float32)
    h = rng.standard_normal((B, S, hidden_size)).astype(np.float32)

    emb_per_layer_w_bf = mx.array(emb_per_layer_w).astype(mx.bfloat16)
    proj_w_bf = mx.array(proj_w).astype(mx.bfloat16)
    proj_norm_w_bf = mx.array(proj_norm_w).astype(mx.bfloat16)
    h_bf = mx.array(h).astype(mx.bfloat16)
    ids_mx = mx.array(ids)

    flat = ids_mx.reshape(-1)
    ple_emb = emb_per_layer_w_bf[flat]
    ple_emb = ple_emb.reshape(B, S, num_layers * ple_dim)
    ple_scaled = ple_emb * mx.array(np.sqrt(ple_dim)).astype(mx.bfloat16)

    h_proj = h_bf @ proj_w_bf.T
    h_scale = mx.array(1.0 / np.sqrt(hidden_size)).astype(mx.bfloat16)
    h_scaled = h_proj * h_scale

    h_reshaped = h_scaled.reshape(B, S, num_layers, ple_dim)
    ple_reshaped = ple_scaled.reshape(B, S, num_layers, ple_dim)

    normed = mx.fast.rms_norm(h_reshaped, proj_norm_w_bf, eps)
    combined = (normed + ple_reshaped) * mx.array(1.0 / np.sqrt(2.0)).astype(mx.bfloat16)

    result = combined
    result_f32 = np.array(result.astype(mx.float32))
    h_f32 = np.array(h_bf.astype(mx.float32))
    emb_f32 = np.array(emb_per_layer_w_bf.astype(mx.float32))
    proj_f32 = np.array(proj_w_bf.astype(mx.float32))

    n = B * S * num_layers * ple_dim
    print(f"#define PLE_INPUTS_B {B}")
    print(f"#define PLE_INPUTS_S {S}")
    print(f"#define PLE_INPUTS_HIDDEN {hidden_size}")
    print(f"#define PLE_INPUTS_PLE_DIM {ple_dim}")
    print(f"#define PLE_INPUTS_LAYERS {num_layers}")
    print(f"#define PLE_INPUTS_VOCAB 3")
    print(f"#define PLE_INPUTS_N {n}")
    print(fmt_c_array("ple_inputs_h", h_f32))
    print(fmt_c_array("ple_inputs_emb_w", emb_f32))
    print(fmt_c_array("ple_inputs_proj_w", proj_f32))
    print(fmt_c_array("ple_inputs_expected", result_f32))
    print()


def gen_ple_apply():
    """Golden for fwd_ple_apply: reimplements the DecoderLayer PLE residual path.

    Faithful reimplementation of the mlx-lm gemma4_text DecoderLayer PLE math
    (gate -> gelu -> multiply -> project -> norm -> residual) using raw mlx
    primitives - not a live call into mlx_lm modules.
    """
    print("/* --- PLE apply golden (per-layer gate/project/norm/residual) --- */")

    B, S = 1, 2
    hidden_size = 8
    ple_dim = 4
    num_layers = 2
    layer = 0
    eps = 1e-6

    rng = np.random.default_rng(101)

    ple_inputs_data = rng.standard_normal((B, S, num_layers, ple_dim)).astype(np.float32)
    h_data = rng.standard_normal((B, S, hidden_size)).astype(np.float32)
    gate_w = rng.standard_normal((ple_dim, hidden_size)).astype(np.float32)
    proj_w = rng.standard_normal((hidden_size, ple_dim)).astype(np.float32)
    norm_w = np.ones(hidden_size, dtype=np.float32)

    ple_inputs_bf = mx.array(ple_inputs_data).astype(mx.bfloat16)
    h_bf = mx.array(h_data).astype(mx.bfloat16)
    gate_w_bf = mx.array(gate_w).astype(mx.bfloat16)
    proj_w_bf = mx.array(proj_w).astype(mx.bfloat16)
    norm_w_bf = mx.array(norm_w).astype(mx.bfloat16)

    sliced = ple_inputs_bf[:, :, layer, :]
    gate_in = h_bf @ gate_w_bf.T
    gate_act = nn.gelu_approx(gate_in)
    gated = sliced * gate_act
    projected = gated @ proj_w_bf.T
    normed = mx.fast.rms_norm(projected, norm_w_bf, eps)
    result = h_bf + normed

    result_f32 = np.array(result.astype(mx.float32))
    h_f32 = np.array(h_bf.astype(mx.float32))
    ple_f32 = np.array(ple_inputs_bf.astype(mx.float32))
    gate_w_f32 = np.array(gate_w_bf.astype(mx.float32))
    proj_w_f32 = np.array(proj_w_bf.astype(mx.float32))

    n_result = B * S * hidden_size
    n_ple = B * S * num_layers * ple_dim
    print(f"#define PLE_APPLY_B {B}")
    print(f"#define PLE_APPLY_S {S}")
    print(f"#define PLE_APPLY_HIDDEN {hidden_size}")
    print(f"#define PLE_APPLY_PLE_DIM {ple_dim}")
    print(f"#define PLE_APPLY_LAYERS {num_layers}")
    print(f"#define PLE_APPLY_LAYER {layer}")
    print(f"#define PLE_APPLY_N {n_result}")
    print(f"#define PLE_APPLY_PLE_N {n_ple}")
    print(fmt_c_array("ple_apply_h", h_f32))
    print(fmt_c_array("ple_apply_ple_inputs", ple_f32))
    print(fmt_c_array("ple_apply_gate_w", gate_w_f32))
    print(fmt_c_array("ple_apply_proj_w", proj_w_f32))
    print(fmt_c_array("ple_apply_expected", result_f32))
    print()


if __name__ == "__main__":
    gen_gelu_approx()
    gen_proportional_rope()
    gen_softcap()
    gen_ple_inputs()
    gen_ple_apply()
