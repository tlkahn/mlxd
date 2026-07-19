#!/usr/bin/env python3
"""Dev-time oracle for gemma4 GELU approx and proportional RoPE goldens.

Generates ground-truth C float array literals from mlx-lm / mlx primitives.
Oracle output is ground truth: a C-side mismatch is a C bug; vectors are
never edited to fit.

Run from the repo root:
    uv run --with mlx python tests/tools/gen_gemma4_goldens.py
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np


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

    Two cases matching the plan:
    1) fixture-shaped: partial_rotary=0.5, factor=8.0, head_dim=16
    2) E2B-shaped: partial_rotary=0.25, factor=1.0, head_dim=16
    """
    print("/* --- Proportional RoPE golden (manual, matching mlx-lm rope_utils) --- */")

    for case_name, partial_rotary, factor, base in [
        ("fixture", 0.5, 8.0, 1000000.0),
        ("e2b", 0.25, 1.0, 1000000.0),
    ]:
        head_dim = 16
        rotated_dims = int(partial_rotary * head_dim)
        n_freqs = rotated_dims // 2
        H = 2
        S = 3
        offset = 2

        # Compute freqs the same way mlxd does: base^(2i/d) * factor
        freqs = np.array(
            [base ** (2.0 * i / rotated_dims) * factor for i in range(n_freqs)],
            dtype=np.float32,
        )

        # Build deterministic bf16 input [1, H, S, head_dim]
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((1, H, S, head_dim)).astype(np.float32)
        x_bf16 = mx.array(x_np).astype(mx.bfloat16)

        # Apply rope via mlx_fast_rope
        freqs_mx = mx.array(freqs)
        out = mx.fast.rope(
            x_bf16,
            rotated_dims,
            traditional=False,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=freqs_mx,
        )
        out_f32 = np.array(out.astype(mx.float32))
        in_f32 = np.array(x_bf16.astype(mx.float32))

        tag = case_name
        print(f"/* Case: {case_name} (partial_rotary={partial_rotary}, factor={factor}) */")
        print(f"#define ROPE_{tag.upper()}_H {H}")
        print(f"#define ROPE_{tag.upper()}_S {S}")
        print(f"#define ROPE_{tag.upper()}_HD {head_dim}")
        print(f"#define ROPE_{tag.upper()}_DIMS {rotated_dims}")
        print(f"#define ROPE_{tag.upper()}_OFFSET {offset}")
        print(f"#define ROPE_{tag.upper()}_N {1 * H * S * head_dim}")
        print(f"#define ROPE_{tag.upper()}_NFREQS {n_freqs}")
        print(fmt_c_array(f"rope_{tag}_freqs", freqs))
        print(fmt_c_array(f"rope_{tag}_input", in_f32))
        print(fmt_c_array(f"rope_{tag}_expected", out_f32))
        print()


if __name__ == "__main__":
    gen_gelu_approx()
    gen_proportional_rope()
