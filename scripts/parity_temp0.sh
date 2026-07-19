#!/bin/sh
# Temp-0 TEXT parity smoke test: compares decoded text output between
# mlx-serve (oracle) and mlxd at temperature 0. A true token-id parity
# gate is deferred because the oracle cannot emit raw token IDs.
set -eu

CKPT="${1:-}"
PROMPT="${2:-The capital of France is}"
PORT="${MLXD_PARITY_PORT:-18653}"
MAX_TOKENS="${MLXD_PARITY_MAX_TOKENS:-50}"
# Mode: "raw" (default) hits /v1/completions + mlxd --raw.
#        "chat" hits /v1/chat/completions + mlxd with chat template (--no-think).
# Chat mode is the right gate for instruct checkpoints (gemma4-it etc.) whose
# raw completion path collapses into short loops that trip mlx-serve's
# degenerate-loop detector and produce length-only mismatches.
MODE="${MLXD_PARITY_MODE:-raw}"

# --- Skip branch: never fails CI ---

if [ -z "${MLXD_MLX_SERVE_BIN:-}" ]; then
    echo "skipped: MLXD_MLX_SERVE_BIN not set"
    exit 0
fi

if ! command -v "$MLXD_MLX_SERVE_BIN" >/dev/null 2>&1 && [ ! -x "$MLXD_MLX_SERVE_BIN" ]; then
    echo "skipped: $MLXD_MLX_SERVE_BIN not executable"
    exit 0
fi

if [ -z "$CKPT" ] || [ ! -d "$CKPT" ]; then
    echo "skipped: checkpoint dir '${CKPT}' missing or not provided"
    exit 0
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "skipped: python3 not found"
    exit 0
fi

if ! command -v curl >/dev/null 2>&1; then
    echo "skipped: curl not found"
    exit 0
fi

# --- Build mlxd if absent ---

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "building mlxd..."
if ! make -C "$REPO_DIR" mlxd >/dev/null 2>&1; then
    echo "FAIL: mlxd build failed" >&2
    exit 1
fi

# --- Fail fast if port is already in use ---

if python3 -c 'import socket,sys; s=socket.socket(); s.settimeout(0.5); sys.exit(0 if s.connect_ex(("127.0.0.1",int(sys.argv[1])))==0 else 1)' "$PORT" 2>/dev/null; then
    echo "FAIL: port $PORT already in use; set MLXD_PARITY_PORT to a free port" >&2
    exit 1
fi

# --- Start oracle ---

WORK=$(mktemp -d)
ORACLE_PID=""

stop_oracle() {
    if [ -n "${ORACLE_PID}" ]; then
        kill "$ORACLE_PID" 2>/dev/null || true
        _w=0
        while kill -0 "$ORACLE_PID" 2>/dev/null && [ "$_w" -lt 5 ]; do
            sleep 1
            _w=$((_w + 1))
        done
        if kill -0 "$ORACLE_PID" 2>/dev/null; then
            kill -9 "$ORACLE_PID" 2>/dev/null || true
        fi
        wait "$ORACLE_PID" 2>/dev/null || true
        ORACLE_PID=""
    fi
}

cleanup() {
    stop_oracle
    rm -rf "$WORK"
}
trap cleanup EXIT

# MLXD_MLX_SERVE_EXTRA_ARGS: optional extra oracle flags (e.g. --skip-mem-preflight).
# Word-split intentionally - callers pass simple flag tokens only.
# shellcheck disable=SC2086
echo "oracle flags: --no-pld --no-mtp ${MLXD_MLX_SERVE_EXTRA_ARGS:-}"
"$MLXD_MLX_SERVE_BIN" --model "$CKPT" --serve --port "$PORT" --no-pld --no-mtp ${MLXD_MLX_SERVE_EXTRA_ARGS:-} >/dev/null 2>&1 &
ORACLE_PID=$!

echo "waiting for oracle on port $PORT..."
TRIES=0
while [ "$TRIES" -lt 120 ]; do
    if ! kill -0 "$ORACLE_PID" 2>/dev/null; then
        echo "FAIL: oracle exited before becoming healthy" >&2
        exit 1
    fi
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        break
    fi
    sleep 1
    TRIES=$((TRIES + 1))
done

if [ "$TRIES" -ge 120 ]; then
    echo "FAIL: oracle did not become healthy within 120s"
    exit 1
fi

# --- Oracle text (via temp file, no command substitution) ---

echo "parity mode: $MODE"
case "$MODE" in
    raw)
        BODY=$(python3 -c 'import json,sys; print(json.dumps({"prompt": sys.argv[1], "temperature": 0, "max_tokens": int(sys.argv[2]), "stream": False}))' "$PROMPT" "$MAX_TOKENS")
        ORACLE_PATH="/v1/completions"
        EXTRACT='import sys,json; d=json.load(open(sys.argv[1],"rb")); open(sys.argv[2],"wb").write(d["choices"][0]["text"].encode("utf-8"))'
        MLXD_EXTRA="--raw"
        ;;
    chat)
        BODY=$(python3 -c 'import json,sys; print(json.dumps({"messages":[{"role":"user","content":sys.argv[1]}], "temperature": 0, "max_tokens": int(sys.argv[2]), "stream": False}))' "$PROMPT" "$MAX_TOKENS")
        ORACLE_PATH="/v1/chat/completions"
        EXTRACT='import sys,json; d=json.load(open(sys.argv[1],"rb")); open(sys.argv[2],"wb").write(d["choices"][0]["message"]["content"].encode("utf-8"))'
        # --no-think keeps enable_thinking=false so the rendered template matches
        # the oracle's default non-thinking chat path for gemma4.
        MLXD_EXTRA="--no-think"
        ;;
    *)
        echo "FAIL: unknown MLXD_PARITY_MODE='$MODE' (want raw|chat)" >&2
        exit 1
        ;;
esac

if ! curl -sf -X POST "http://127.0.0.1:$PORT$ORACLE_PATH" \
    -H "Content-Type: application/json" \
    -d "$BODY" \
    -o "$WORK/resp.json"; then
    echo "FAIL: oracle $ORACLE_PATH request failed" >&2
    exit 1
fi
if ! python3 -c "$EXTRACT" \
    "$WORK/resp.json" "$WORK/oracle.txt" 2>/dev/null; then
    echo "FAIL: could not parse oracle response" >&2
    exit 1
fi

stop_oracle

# --- mlxd text (via temp file, no command substitution) ---
# shellcheck disable=SC2086
"$REPO_DIR/mlxd" run "$CKPT" "$PROMPT" $MLXD_EXTRA --temperature 0 --max-tokens "$MAX_TOKENS" \
    > "$WORK/mlxd.txt"

# --- Compare (byte-exact via parity_compare.py) ---

if python3 "$SCRIPT_DIR/parity_compare.py" "$WORK/oracle.txt" "$WORK/mlxd.txt"; then
    echo "text parity OK (mode=$MODE, max_tokens=$MAX_TOKENS)"
    exit 0
else
    echo ""
    echo "--- oracle text ---"
    cat "$WORK/oracle.txt"; echo
    echo "--- mlxd text ---"
    cat "$WORK/mlxd.txt"
    echo "--- mlxd token ids ---"
    # shellcheck disable=SC2086
    "$REPO_DIR/mlxd" run "$CKPT" "$PROMPT" $MLXD_EXTRA --temperature 0 --max-tokens "$MAX_TOKENS" --token-ids
    exit 1
fi
