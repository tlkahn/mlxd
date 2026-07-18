#!/bin/sh
set -eu

CKPT="${1:-}"
PROMPT="${2:-The capital of France is}"
PORT="${MLXD_PARITY_PORT:-18653}"
MAX_TOKENS=50

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

if [ ! -x "$REPO_DIR/mlxd" ]; then
    echo "building mlxd..."
    if ! make -C "$REPO_DIR" mlxd >/dev/null; then
        echo "FAIL: mlxd build failed" >&2
        exit 1
    fi
fi

# --- Start oracle ---

cleanup() {
    if [ -n "${ORACLE_PID:-}" ]; then
        kill "$ORACLE_PID" 2>/dev/null || true
        wait "$ORACLE_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

"$MLXD_MLX_SERVE_BIN" --model "$CKPT" --serve --port "$PORT" >/dev/null 2>&1 &
ORACLE_PID=$!

echo "waiting for oracle on port $PORT..."
TRIES=0
while [ "$TRIES" -lt 120 ]; do
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

# --- Oracle tokens ---

ORACLE_TEXT=$(curl -sf -X POST "http://127.0.0.1:$PORT/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"prompt\": \"$PROMPT\", \"temperature\": 0, \"max_tokens\": $MAX_TOKENS, \"stream\": false}" \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['text'], end='')")

# --- mlxd tokens ---

MLXD_TEXT=$("$REPO_DIR/mlxd" run "$CKPT" "$PROMPT" --raw --temperature 0 --max-tokens "$MAX_TOKENS")

# --- Compare ---

if [ "$ORACLE_TEXT" = "$MLXD_TEXT" ]; then
    echo "parity OK ($MAX_TOKENS tokens)"
    exit 0
else
    echo "MISMATCH"
    echo "oracle: $ORACLE_TEXT"
    echo "mlxd:   $MLXD_TEXT"
    echo ""
    echo "--- mlxd token ids ---"
    "$REPO_DIR/mlxd" run "$CKPT" "$PROMPT" --raw --temperature 0 --max-tokens "$MAX_TOKENS" --token-ids
    exit 1
fi
