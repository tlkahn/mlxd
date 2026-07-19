#!/bin/sh
# parity_family.sh - family-parameterized wrapper around parity_temp0.sh
#
# Usage: parity_family.sh <family>|all [prompt]
#
# Resolves a per-family HuggingFace checkpoint path and delegates to
# parity_temp0.sh.  Skips gracefully when prerequisites are absent (never
# fails CI).
set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

usage() {
    printf 'usage: parity_family.sh <family>|all [prompt]\n' >&2
    printf 'families: qwen3 gemma3 qwen2 llama mistral lfm2\n' >&2
    exit 2
}

# Family -> canonical HF checkpoint id table
# Empty string = TBD (D1-D6 fill in)
family_id() {
    case "$1" in
        qwen3)   echo "mlx-community/Qwen3-0.6B-4bit" ;;
        gemma3)  echo "" ;;
        qwen2)   echo "" ;;
        llama)   echo "" ;;
        mistral) echo "" ;;
        lfm2)    echo "" ;;
        *)       return 1 ;;
    esac
}

ALL_FAMILIES="qwen3 gemma3 qwen2 llama mistral lfm2"

[ $# -lt 1 ] && usage

FAMILY="$1"
PROMPT="${2:-Hello}"

run_one() {
    _fam="$1"
    _prompt="$2"

    _id=$(family_id "$_fam") || { usage; }

    if [ -z "$_id" ]; then
        printf 'skipped: no canonical checkpoint id for %s\n' "$_fam"
        return 0
    fi

    # Per-family env override: MLXD_PARITY_CKPT_<FAMILY_UPPER>
    _fam_upper=$(printf '%s' "$_fam" | tr '[:lower:]' '[:upper:]')
    _override_var="MLXD_PARITY_CKPT_${_fam_upper}"
    eval "_override=\${${_override_var}:-}"

    if [ -n "$_override" ]; then
        _dir="$_override"
    elif [ -n "${MLXD_PARITY_CKPT_ROOT:-}" ]; then
        _dir="${MLXD_PARITY_CKPT_ROOT}/${_id}"
    else
        printf 'skipped: no checkpoint root set for %s\n' "$_fam"
        return 0
    fi

    if [ ! -d "$_dir" ]; then
        printf 'skipped: checkpoint dir absent for %s (%s)\n' "$_fam" "$_dir"
        return 0
    fi

    "$SCRIPT_DIR/parity_temp0.sh" "$_dir" "$_prompt"
}

if [ "$FAMILY" = "all" ]; then
    _any_fail=0
    for _f in $ALL_FAMILIES; do
        printf '%s: ' "$_f"
        run_one "$_f" "$PROMPT" || _any_fail=1
    done
    exit "$_any_fail"
fi

# Single family
family_id "$FAMILY" >/dev/null 2>&1 || usage
run_one "$FAMILY" "$PROMPT"
