#!/bin/sh
# Offline tests for scripts/parity_temp0.sh and scripts/parity_compare.py.
# No model, no network beyond loopback. Run via `make test-parity-script`.
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPARE="$REPO_DIR/scripts/parity_compare.py"

FAILS=0

pass() { printf "  %-40sOK\n" "$1"; }
fail() { printf "  %-40sFAIL (%s)\n" "$1" "$2"; FAILS=$((FAILS + 1)); }

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# --- compare: identical files match ---

printf 'hello world' > "$TMP/oracle.txt"
printf 'hello world' > "$TMP/mlxd.txt"
if python3 "$COMPARE" "$TMP/oracle.txt" "$TMP/mlxd.txt" >/dev/null 2>&1; then
    pass "compare-identical"
else
    fail "compare-identical" "expected exit 0"
fi

# --- compare: mlxd trailing newline stripped ---

printf 'hello world' > "$TMP/oracle.txt"
printf 'hello world\n' > "$TMP/mlxd.txt"
if python3 "$COMPARE" "$TMP/oracle.txt" "$TMP/mlxd.txt" >/dev/null 2>&1; then
    pass "compare-strip-newline"
else
    fail "compare-strip-newline" "expected exit 0"
fi

# --- compare: differing content -> MISMATCH ---

printf 'hello world' > "$TMP/oracle.txt"
printf 'hello earth' > "$TMP/mlxd.txt"
out=$(python3 "$COMPARE" "$TMP/oracle.txt" "$TMP/mlxd.txt" 2>&1) && rc=0 || rc=$?
if [ "$rc" -ne 0 ] && printf '%s\n' "$out" | grep -q 'MISMATCH'; then
    pass "compare-mismatch"
else
    fail "compare-mismatch" "expected exit 1 + MISMATCH"
fi

# --- compare: regression - oracle trailing \n vs mlxd without -> MISMATCH ---

printf 'x\n' > "$TMP/oracle.txt"
printf 'x' > "$TMP/mlxd.txt"
out=$(python3 "$COMPARE" "$TMP/oracle.txt" "$TMP/mlxd.txt" 2>&1) && rc=0 || rc=$?
if [ "$rc" -ne 0 ] && printf '%s\n' "$out" | grep -q 'MISMATCH'; then
    pass "compare-regression-trailing-nl"
else
    fail "compare-regression-trailing-nl" "expected MISMATCH (masked by old $(..)-compare)"
fi

# --- rebuild: stale binary triggers make (finding 3) ---
# A Makefile with a failing target proves make was invoked even
# when an executable mlxd already exists (old code skipped the build).

REBUILD_DIR=$(mktemp -d)
mkdir -p "$REBUILD_DIR/scripts" "$REBUILD_DIR/ckpt"
cp "$REPO_DIR/scripts/parity_temp0.sh" "$REBUILD_DIR/scripts/"
printf '#!/bin/sh\nexit 0\n' > "$REBUILD_DIR/stub"; chmod +x "$REBUILD_DIR/stub"
printf '#!/bin/sh\nexit 0\n' > "$REBUILD_DIR/mlxd"; chmod +x "$REBUILD_DIR/mlxd"
printf '.PHONY: mlxd\nmlxd:\n\t@false\n' > "$REBUILD_DIR/Makefile"

MLXD_MLX_SERVE_BIN="$REBUILD_DIR/stub" sh "$REBUILD_DIR/scripts/parity_temp0.sh" "$REBUILD_DIR/ckpt" >"$TMP/rebuild.out" 2>&1 &
WPID=$!
ELAPSED=0
while kill -0 "$WPID" 2>/dev/null && [ "$ELAPSED" -lt 15 ]; do
    sleep 1
    ELAPSED=$((ELAPSED + 1))
done
if kill -0 "$WPID" 2>/dev/null; then
    kill "$WPID" 2>/dev/null; wait "$WPID" 2>/dev/null || true
fi
wait "$WPID" 2>/dev/null && rc=0 || rc=$?
rm -rf "$REBUILD_DIR"

if [ "$rc" -ne 0 ] && grep -q 'build failed' "$TMP/rebuild.out"; then
    pass "rebuild-stale-binary"
else
    fail "rebuild-stale-binary" "expected nonzero + 'build failed' (rc=$rc)"
fi

# --- busy-port: fail fast when port already in use (finding 4) ---

BPORT_DIR=$(mktemp -d)
mkdir -p "$BPORT_DIR/scripts" "$BPORT_DIR/ckpt"
cp "$REPO_DIR/scripts/parity_temp0.sh" "$BPORT_DIR/scripts/"
printf '#!/bin/sh\nexit 0\n' > "$BPORT_DIR/stub"; chmod +x "$BPORT_DIR/stub"
printf '.PHONY: mlxd\nmlxd:\n\t@true\n' > "$BPORT_DIR/Makefile"

# Start a listener on an ephemeral port
python3 -c '
import socket, sys, time
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("127.0.0.1", 0))
port = s.getsockname()[1]
s.listen(1)
sys.stdout.write(str(port))
sys.stdout.flush()
time.sleep(30)
' >"$TMP/bport.port" &
LISTENER_PID=$!
sleep 1
BPORT=$(cat "$TMP/bport.port")

MLXD_MLX_SERVE_BIN="$BPORT_DIR/stub" MLXD_PARITY_PORT="$BPORT" \
    sh "$BPORT_DIR/scripts/parity_temp0.sh" "$BPORT_DIR/ckpt" >"$TMP/bport.out" 2>&1 &
SPID=$!
ELAPSED=0
while kill -0 "$SPID" 2>/dev/null && [ "$ELAPSED" -lt 10 ]; do
    sleep 1
    ELAPSED=$((ELAPSED + 1))
done
if kill -0 "$SPID" 2>/dev/null; then
    kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null || true
fi
wait "$SPID" 2>/dev/null && rc=0 || rc=$?
kill "$LISTENER_PID" 2>/dev/null; wait "$LISTENER_PID" 2>/dev/null || true
rm -rf "$BPORT_DIR"

if [ "$rc" -ne 0 ] && grep -q 'already in use' "$TMP/bport.out"; then
    pass "busy-port-fail-fast"
else
    fail "busy-port-fail-fast" "expected nonzero + 'already in use' (rc=$rc)"
fi

# --- e2e helper: set up a stub oracle + fake mlxd environment ---
# Usage: setup_e2e <dir> <oracle_text> <mlxd_text> [mlxd_token_ids]
setup_e2e() {
    _dir="$1"; _oracle_text="$2"; _mlxd_text="$3"; _token_ids="${4:-1 2 3}"
    mkdir -p "$_dir/scripts" "$_dir/ckpt"
    cp "$REPO_DIR/scripts/parity_temp0.sh" "$_dir/scripts/"
    cp "$REPO_DIR/scripts/parity_compare.py" "$_dir/scripts/"
    printf '.PHONY: mlxd\nmlxd:\n\t@true\n' > "$_dir/Makefile"

    # Write oracle text to file (preserves exact bytes, no shell stripping)
    printf '%s' "$_oracle_text" > "$_dir/oracle_text.dat"

    # Stub oracle: HTTP server answering /health and /v1/completions
    cat > "$_dir/stub_oracle.py" <<'PYEOF'
import http.server, json, os, sys

class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
    def do_POST(self):
        if self.path == "/v1/completions":
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            text = open(os.environ["STUB_ORACLE_TEXT_FILE"]).read()
            body = json.dumps({"choices": [{"text": text}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    def log_message(self, *a): pass

port = int(sys.argv[1])
srv = http.server.HTTPServer(("127.0.0.1", port), H)
srv.serve_forever()
PYEOF
    # Stub oracle launcher (mimics mlx-serve CLI: --model X --serve --port N)
    cat > "$_dir/stub_oracle_bin" <<SHEOF
#!/bin/sh
printf '%s\n' "\$@" > "$_dir/oracle_args.txt"
exec python3 "$_dir/stub_oracle.py" "\$5"
SHEOF
    chmod +x "$_dir/stub_oracle_bin"

    # Fake mlxd: prints fixed text (+ trailing newline as real mlxd does),
    # or token IDs when --token-ids is present.
    cat > "$_dir/mlxd" <<SHEOF
#!/bin/sh
for arg in "\$@"; do
    if [ "\$arg" = "--token-ids" ]; then
        printf '%s\n' "$_token_ids"
        exit 0
    fi
done
printf '%s\n' "$_mlxd_text"
SHEOF
    chmod +x "$_dir/mlxd"
}

# Pick a free port
pick_free_port() {
    python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()'
}

# --- e2e: matching text -> text parity OK (cycle 7) ---

E2E_MATCH_DIR=$(mktemp -d)
E2E_PORT=$(pick_free_port)
setup_e2e "$E2E_MATCH_DIR" "hello world" "hello world"

STUB_ORACLE_TEXT_FILE="$E2E_MATCH_DIR/oracle_text.dat" MLXD_MLX_SERVE_BIN="$E2E_MATCH_DIR/stub_oracle_bin" \
    MLXD_PARITY_PORT="$E2E_PORT" \
    sh "$E2E_MATCH_DIR/scripts/parity_temp0.sh" "$E2E_MATCH_DIR/ckpt" >"$TMP/e2e_match.out" 2>&1 &
E2E_PID=$!
ELAPSED=0
while kill -0 "$E2E_PID" 2>/dev/null && [ "$ELAPSED" -lt 15 ]; do
    sleep 1
    ELAPSED=$((ELAPSED + 1))
done
if kill -0 "$E2E_PID" 2>/dev/null; then
    kill "$E2E_PID" 2>/dev/null; wait "$E2E_PID" 2>/dev/null || true
fi
wait "$E2E_PID" 2>/dev/null && rc=0 || rc=$?
cp "$E2E_MATCH_DIR/oracle_args.txt" "$TMP/e2e_match_args.txt" 2>/dev/null || true
rm -rf "$E2E_MATCH_DIR"

if [ "$rc" -eq 0 ] && grep -q 'text parity OK' "$TMP/e2e_match.out" && \
   grep -q -- '--no-pld' "$TMP/e2e_match_args.txt" && \
   grep -q -- '--no-mtp' "$TMP/e2e_match_args.txt"; then
    pass "e2e-match"
else
    fail "e2e-match" "expected exit 0 + 'text parity OK' + pinned flags (rc=$rc)"
fi

# --- e2e: masked divergence -> MISMATCH (cycle 8) ---
# Oracle emits "hello\n" as text, mlxd prints "hello\n" (content + presentation newline).
# Old command-substitution compare masked this; the new file-based compare catches it.

E2E_MASK_DIR=$(mktemp -d)
E2E_PORT2=$(pick_free_port)
setup_e2e "$E2E_MASK_DIR" "hello" "hello"
# Overwrite oracle_text.dat with "hello\n" (trailing newline that setup_e2e
# can't pass through shell arguments without command-substitution stripping)
printf 'hello\n' > "$E2E_MASK_DIR/oracle_text.dat"

STUB_ORACLE_TEXT_FILE="$E2E_MASK_DIR/oracle_text.dat" MLXD_MLX_SERVE_BIN="$E2E_MASK_DIR/stub_oracle_bin" \
    MLXD_PARITY_PORT="$E2E_PORT2" \
    sh "$E2E_MASK_DIR/scripts/parity_temp0.sh" "$E2E_MASK_DIR/ckpt" >"$TMP/e2e_mask.out" 2>&1 &
E2E_PID2=$!
ELAPSED=0
while kill -0 "$E2E_PID2" 2>/dev/null && [ "$ELAPSED" -lt 15 ]; do
    sleep 1
    ELAPSED=$((ELAPSED + 1))
done
if kill -0 "$E2E_PID2" 2>/dev/null; then
    kill "$E2E_PID2" 2>/dev/null; wait "$E2E_PID2" 2>/dev/null || true
fi
wait "$E2E_PID2" 2>/dev/null && rc=0 || rc=$?
rm -rf "$E2E_MASK_DIR"

if [ "$rc" -ne 0 ] && grep -q 'MISMATCH' "$TMP/e2e_mask.out"; then
    pass "e2e-masked-divergence"
else
    fail "e2e-masked-divergence" "expected exit 1 + MISMATCH (rc=$rc)"
fi

# --- e2e: oracle crash -> fail fast (cycle C) ---

E2E_CRASH_DIR=$(mktemp -d)
E2E_CRASH_PORT=$(pick_free_port)
setup_e2e "$E2E_CRASH_DIR" "hello" "hello"

# Overwrite oracle with one that exits immediately
cat > "$E2E_CRASH_DIR/stub_oracle_bin" <<'SHEOF'
#!/bin/sh
exit 1
SHEOF
chmod +x "$E2E_CRASH_DIR/stub_oracle_bin"

STUB_ORACLE_TEXT_FILE="$E2E_CRASH_DIR/oracle_text.dat" MLXD_MLX_SERVE_BIN="$E2E_CRASH_DIR/stub_oracle_bin" \
    MLXD_PARITY_PORT="$E2E_CRASH_PORT" \
    sh "$E2E_CRASH_DIR/scripts/parity_temp0.sh" "$E2E_CRASH_DIR/ckpt" >"$TMP/e2e_crash.out" 2>&1 &
E2E_CRASH_PID=$!
ELAPSED=0
while kill -0 "$E2E_CRASH_PID" 2>/dev/null && [ "$ELAPSED" -lt 15 ]; do
    sleep 1
    ELAPSED=$((ELAPSED + 1))
done
if kill -0 "$E2E_CRASH_PID" 2>/dev/null; then
    kill "$E2E_CRASH_PID" 2>/dev/null; wait "$E2E_CRASH_PID" 2>/dev/null || true
fi
wait "$E2E_CRASH_PID" 2>/dev/null && rc=0 || rc=$?
rm -rf "$E2E_CRASH_DIR"

if [ "$rc" -ne 0 ] && grep -q 'oracle exited before becoming healthy' "$TMP/e2e_crash.out"; then
    pass "e2e-oracle-crash"
else
    fail "e2e-oracle-crash" "expected nonzero + 'oracle exited' (rc=$rc)"
fi

# --- e2e: oracle 500 -> diagnostic message (finding 9 residual) ---

E2E_500_DIR=$(mktemp -d)
E2E_500_PORT=$(pick_free_port)
setup_e2e "$E2E_500_DIR" "hello" "hello"

# Overwrite oracle to return 500 on completions (health stays 200)
cat > "$E2E_500_DIR/stub_oracle.py" <<'PYEOF'
import http.server, sys

class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
    def do_POST(self):
        if self.path == "/v1/completions":
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            self.send_response(500)
            self.end_headers()
    def log_message(self, *a): pass

port = int(sys.argv[1])
srv = http.server.HTTPServer(("127.0.0.1", port), H)
srv.serve_forever()
PYEOF

STUB_ORACLE_TEXT_FILE="$E2E_500_DIR/oracle_text.dat" MLXD_MLX_SERVE_BIN="$E2E_500_DIR/stub_oracle_bin" \
    MLXD_PARITY_PORT="$E2E_500_PORT" \
    sh "$E2E_500_DIR/scripts/parity_temp0.sh" "$E2E_500_DIR/ckpt" >"$TMP/e2e_500.out" 2>&1 &
E2E_500_PID=$!
ELAPSED=0
while kill -0 "$E2E_500_PID" 2>/dev/null && [ "$ELAPSED" -lt 15 ]; do
    sleep 1
    ELAPSED=$((ELAPSED + 1))
done
if kill -0 "$E2E_500_PID" 2>/dev/null; then
    kill "$E2E_500_PID" 2>/dev/null; wait "$E2E_500_PID" 2>/dev/null || true
fi
wait "$E2E_500_PID" 2>/dev/null && rc=0 || rc=$?
rm -rf "$E2E_500_DIR"

if [ "$rc" -ne 0 ] && grep -q 'FAIL: oracle completion request failed' "$TMP/e2e_500.out"; then
    pass "e2e-oracle-500"
else
    fail "e2e-oracle-500" "expected nonzero + 'oracle completion request failed' (rc=$rc)"
fi

# --- e2e: oracle bad JSON -> diagnostic message (finding 9 residual) ---

E2E_BADJSON_DIR=$(mktemp -d)
E2E_BADJSON_PORT=$(pick_free_port)
setup_e2e "$E2E_BADJSON_DIR" "hello" "hello"

# Overwrite oracle to return 200 with non-JSON body
cat > "$E2E_BADJSON_DIR/stub_oracle.py" <<'PYEOF'
import http.server, sys

class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
    def do_POST(self):
        if self.path == "/v1/completions":
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", "8")
            self.end_headers()
            self.wfile.write(b"not json")
    def log_message(self, *a): pass

port = int(sys.argv[1])
srv = http.server.HTTPServer(("127.0.0.1", port), H)
srv.serve_forever()
PYEOF

STUB_ORACLE_TEXT_FILE="$E2E_BADJSON_DIR/oracle_text.dat" MLXD_MLX_SERVE_BIN="$E2E_BADJSON_DIR/stub_oracle_bin" \
    MLXD_PARITY_PORT="$E2E_BADJSON_PORT" \
    sh "$E2E_BADJSON_DIR/scripts/parity_temp0.sh" "$E2E_BADJSON_DIR/ckpt" >"$TMP/e2e_badjson.out" 2>&1 &
E2E_BADJSON_PID=$!
ELAPSED=0
while kill -0 "$E2E_BADJSON_PID" 2>/dev/null && [ "$ELAPSED" -lt 15 ]; do
    sleep 1
    ELAPSED=$((ELAPSED + 1))
done
if kill -0 "$E2E_BADJSON_PID" 2>/dev/null; then
    kill "$E2E_BADJSON_PID" 2>/dev/null; wait "$E2E_BADJSON_PID" 2>/dev/null || true
fi
wait "$E2E_BADJSON_PID" 2>/dev/null && rc=0 || rc=$?
rm -rf "$E2E_BADJSON_DIR"

if [ "$rc" -ne 0 ] && grep -q 'FAIL: could not parse oracle response' "$TMP/e2e_badjson.out"; then
    pass "e2e-oracle-bad-json"
else
    fail "e2e-oracle-bad-json" "expected nonzero + 'could not parse oracle response' (rc=$rc)"
fi

# --- e2e: bounded oracle kill + teardown before mlxd (cycle D) ---

E2E_KILL_DIR=$(mktemp -d)
E2E_KILL_PORT=$(pick_free_port)
setup_e2e "$E2E_KILL_DIR" "hello world" "hello world"

# Overwrite oracle Python to ignore SIGTERM and write PID
cat > "$E2E_KILL_DIR/stub_oracle.py" <<'PYEOF'
import http.server, json, os, signal, sys

signal.signal(signal.SIGTERM, signal.SIG_IGN)

pid_file = os.environ.get("STUB_ORACLE_PID_FILE", "")
if pid_file:
    with open(pid_file, "w") as f:
        f.write(str(os.getpid()))

class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
    def do_POST(self):
        if self.path == "/v1/completions":
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            text = open(os.environ["STUB_ORACLE_TEXT_FILE"]).read()
            body = json.dumps({"choices": [{"text": text}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    def log_message(self, *a): pass

port = int(sys.argv[1])
srv = http.server.HTTPServer(("127.0.0.1", port), H)
srv.serve_forever()
PYEOF

# Overwrite mlxd to verify oracle is dead before running
cat > "$E2E_KILL_DIR/mlxd" <<SHEOF
#!/bin/sh
OPID=\$(cat "$E2E_KILL_DIR/oracle_pid.txt" 2>/dev/null || echo "")
if [ -n "\$OPID" ] && kill -0 "\$OPID" 2>/dev/null; then
    echo "FAIL: oracle still alive when mlxd runs" >&2
    exit 1
fi
touch "$E2E_KILL_DIR/mlxd_marker.txt"
for arg in "\$@"; do
    if [ "\$arg" = "--token-ids" ]; then
        printf '%s\n' "1 2 3"
        exit 0
    fi
done
printf '%s\n' "hello world"
SHEOF
chmod +x "$E2E_KILL_DIR/mlxd"

STUB_ORACLE_TEXT_FILE="$E2E_KILL_DIR/oracle_text.dat" \
    STUB_ORACLE_PID_FILE="$E2E_KILL_DIR/oracle_pid.txt" \
    MLXD_MLX_SERVE_BIN="$E2E_KILL_DIR/stub_oracle_bin" \
    MLXD_PARITY_PORT="$E2E_KILL_PORT" \
    sh "$E2E_KILL_DIR/scripts/parity_temp0.sh" "$E2E_KILL_DIR/ckpt" >"$TMP/e2e_kill.out" 2>&1 &
E2E_KILL_PID=$!
ELAPSED=0
while kill -0 "$E2E_KILL_PID" 2>/dev/null && [ "$ELAPSED" -lt 30 ]; do
    sleep 1
    ELAPSED=$((ELAPSED + 1))
done
if kill -0 "$E2E_KILL_PID" 2>/dev/null; then
    kill "$E2E_KILL_PID" 2>/dev/null; wait "$E2E_KILL_PID" 2>/dev/null || true
fi
wait "$E2E_KILL_PID" 2>/dev/null && rc=0 || rc=$?
marker_exists=false
[ -f "$E2E_KILL_DIR/mlxd_marker.txt" ] && marker_exists=true
rm -rf "$E2E_KILL_DIR"

if [ "$rc" -eq 0 ] && [ "$marker_exists" = "true" ]; then
    pass "e2e-bounded-kill"
else
    fail "e2e-bounded-kill" "expected exit 0 + marker (rc=$rc, marker=$marker_exists)"
fi

# ===========================================================================
# parity_family.sh wrapper tests
# ===========================================================================

WRAPPER="$REPO_DIR/scripts/parity_family.sh"

# --- wrapper: unknown family -> exit 2 ---

out=$(sh "$WRAPPER" bogus 2>&1) && rc=0 || rc=$?
if [ "$rc" -eq 2 ] && printf '%s\n' "$out" | grep -qi 'usage'; then
    pass "wrapper-unknown-family"
else
    fail "wrapper-unknown-family" "expected exit 2 + usage (rc=$rc)"
fi

# --- wrapper: id mapping (qwen3 resolves via root) ---

WMAP_DIR=$(mktemp -d)
mkdir -p "$WMAP_DIR/root/mlx-community/Qwen3-0.6B-4bit"
mkdir -p "$WMAP_DIR/scripts"
cp "$WRAPPER" "$WMAP_DIR/scripts/parity_family.sh"
# Stub parity_temp0.sh that records args
cat > "$WMAP_DIR/scripts/parity_temp0.sh" <<'STUBEOF'
#!/bin/sh
printf '%s\n' "$@" > "$(dirname "$0")/../delegate_args.txt"
exit 0
STUBEOF
chmod +x "$WMAP_DIR/scripts/parity_temp0.sh"

MLXD_PARITY_CKPT_ROOT="$WMAP_DIR/root" sh "$WMAP_DIR/scripts/parity_family.sh" qwen3 "hi" >/dev/null 2>&1 && rc=0 || rc=$?
if [ "$rc" -eq 0 ] && [ -f "$WMAP_DIR/delegate_args.txt" ]; then
    arg1=$(sed -n '1p' "$WMAP_DIR/delegate_args.txt")
    arg2=$(sed -n '2p' "$WMAP_DIR/delegate_args.txt")
    if [ "$arg1" = "$WMAP_DIR/root/mlx-community/Qwen3-0.6B-4bit" ] && [ "$arg2" = "hi" ]; then
        pass "wrapper-id-mapping"
    else
        fail "wrapper-id-mapping" "wrong delegate args: $arg1 / $arg2"
    fi
else
    fail "wrapper-id-mapping" "delegate not called (rc=$rc)"
fi
rm -rf "$WMAP_DIR"

# --- wrapper: skip on unset root ---

out=$(env -u MLXD_PARITY_CKPT_ROOT -u MLXD_PARITY_CKPT_QWEN3 sh "$WRAPPER" qwen3 2>&1) && rc=0 || rc=$?
if [ "$rc" -eq 0 ] && printf '%s\n' "$out" | grep -q 'skipped'; then
    pass "wrapper-skip-unset-root"
else
    fail "wrapper-skip-unset-root" "expected exit 0 + skipped (rc=$rc)"
fi

# --- wrapper: skip on missing dir ---

WMISS_DIR=$(mktemp -d)
out=$(MLXD_PARITY_CKPT_ROOT="$WMISS_DIR/nonexistent" sh "$WRAPPER" qwen3 2>&1) && rc=0 || rc=$?
rm -rf "$WMISS_DIR"
if [ "$rc" -eq 0 ] && printf '%s\n' "$out" | grep -q 'skipped'; then
    pass "wrapper-skip-missing-dir"
else
    fail "wrapper-skip-missing-dir" "expected exit 0 + skipped (rc=$rc)"
fi

# --- wrapper: skip on TBD id (gemma3) ---

out=$(MLXD_PARITY_CKPT_ROOT=/tmp sh "$WRAPPER" gemma3 2>&1) && rc=0 || rc=$?
if [ "$rc" -eq 0 ] && printf '%s\n' "$out" | grep -q 'skipped.*no canonical checkpoint id'; then
    pass "wrapper-skip-tbd-id"
else
    fail "wrapper-skip-tbd-id" "expected exit 0 + skipped:no canonical (rc=$rc)"
fi

# --- wrapper: per-family override ---

WOVER_DIR=$(mktemp -d)
mkdir -p "$WOVER_DIR/custom"
mkdir -p "$WOVER_DIR/scripts"
cp "$WRAPPER" "$WOVER_DIR/scripts/parity_family.sh"
cat > "$WOVER_DIR/scripts/parity_temp0.sh" <<'STUBEOF'
#!/bin/sh
printf '%s\n' "$@" > "$(dirname "$0")/../delegate_args.txt"
exit 0
STUBEOF
chmod +x "$WOVER_DIR/scripts/parity_temp0.sh"

MLXD_PARITY_CKPT_QWEN3="$WOVER_DIR/custom" sh "$WOVER_DIR/scripts/parity_family.sh" qwen3 "test" >/dev/null 2>&1 && rc=0 || rc=$?
if [ "$rc" -eq 0 ] && [ -f "$WOVER_DIR/delegate_args.txt" ]; then
    arg1=$(sed -n '1p' "$WOVER_DIR/delegate_args.txt")
    if [ "$arg1" = "$WOVER_DIR/custom" ]; then
        pass "wrapper-per-family-override"
    else
        fail "wrapper-per-family-override" "wrong dir: $arg1"
    fi
else
    fail "wrapper-per-family-override" "delegate not called (rc=$rc)"
fi
rm -rf "$WOVER_DIR"

# --- wrapper: override wins over TBD id (gemma3 has no canonical id) ---

WOVTBD_DIR=$(mktemp -d)
mkdir -p "$WOVTBD_DIR/custom"
mkdir -p "$WOVTBD_DIR/scripts"
cp "$WRAPPER" "$WOVTBD_DIR/scripts/parity_family.sh"
cat > "$WOVTBD_DIR/scripts/parity_temp0.sh" <<'STUBEOF'
#!/bin/sh
printf '%s\n' "$@" > "$(dirname "$0")/../delegate_args.txt"
exit 0
STUBEOF
chmod +x "$WOVTBD_DIR/scripts/parity_temp0.sh"

MLXD_PARITY_CKPT_GEMMA3="$WOVTBD_DIR/custom" sh "$WOVTBD_DIR/scripts/parity_family.sh" gemma3 "test" >/dev/null 2>&1 && rc=0 || rc=$?
if [ "$rc" -eq 0 ] && [ -f "$WOVTBD_DIR/delegate_args.txt" ]; then
    arg1=$(sed -n '1p' "$WOVTBD_DIR/delegate_args.txt")
    if [ "$arg1" = "$WOVTBD_DIR/custom" ]; then
        pass "wrapper-override-tbd-family"
    else
        fail "wrapper-override-tbd-family" "wrong dir: $arg1"
    fi
else
    fail "wrapper-override-tbd-family" "delegate not called (rc=$rc)"
fi
rm -rf "$WOVTBD_DIR"

# --- wrapper: run-all with nothing set -> skip all, exit 0 ---

out=$(env -u MLXD_PARITY_CKPT_ROOT -u MLXD_PARITY_CKPT_QWEN3 -u MLXD_PARITY_CKPT_GEMMA3 -u MLXD_PARITY_CKPT_QWEN2 -u MLXD_PARITY_CKPT_LLAMA -u MLXD_PARITY_CKPT_MISTRAL -u MLXD_PARITY_CKPT_GEMMA4 -u MLXD_PARITY_CKPT_QWEN3_5 sh "$WRAPPER" all 2>&1) && rc=0 || rc=$?
if [ "$rc" -eq 0 ] && printf '%s\n' "$out" | grep -q 'skipped'; then
    pass "wrapper-run-all"
else
    fail "wrapper-run-all" "expected exit 0 + skipped lines (rc=$rc)"
fi

# --- wrapper: lfm2 is now unknown -> exit 2 ---

out=$(sh "$WRAPPER" lfm2 2>&1) && rc=0 || rc=$?
if [ "$rc" -eq 2 ] && printf '%s\n' "$out" | grep -qi 'usage'; then
    pass "wrapper-unknown-family-lfm2"
else
    fail "wrapper-unknown-family-lfm2" "expected exit 2 + usage (rc=$rc)"
fi

# --- wrapper: skip when canonical id present but dir absent (gemma4, qwen3_5) ---

out=$(MLXD_PARITY_CKPT_ROOT=/tmp sh "$WRAPPER" gemma4 2>&1) && rc=0 || rc=$?
if [ "$rc" -eq 0 ] && printf '%s\n' "$out" | grep -q 'skipped.*checkpoint dir absent'; then
    pass "wrapper-skip-absent-gemma4"
else
    fail "wrapper-skip-absent-gemma4" "expected exit 0 + skipped:dir absent (rc=$rc)"
fi

out=$(MLXD_PARITY_CKPT_ROOT=/tmp sh "$WRAPPER" qwen3_5 2>&1) && rc=0 || rc=$?
if [ "$rc" -eq 0 ] && printf '%s\n' "$out" | grep -q 'skipped.*checkpoint dir absent'; then
    pass "wrapper-skip-absent-qwen3_5"
else
    fail "wrapper-skip-absent-qwen3_5" "expected exit 0 + skipped:dir absent (rc=$rc)"
fi

# --- wrapper: qwen3_5 hard-skips even when hybrid snapshot dir is present ---
# Real Qwen3.5 checkpoints are hybrid; Stage D rejects them. Keep the recorded
# canonical id but never invoke parity_temp0.sh via the root path.
W_Q35_DIR=$(mktemp -d)
mkdir -p "$W_Q35_DIR/root/mlx-community/Qwen3.5-0.8B-4bit"
out=$(MLXD_PARITY_CKPT_ROOT="$W_Q35_DIR/root" sh "$WRAPPER" qwen3_5 2>&1) && rc=0 || rc=$?
rm -rf "$W_Q35_DIR"
if [ "$rc" -eq 0 ] && printf '%s\n' "$out" | grep -q 'skipped: qwen3_5 hybrid parity deferred to Stage E'; then
    pass "wrapper-skip-present-qwen3_5-hybrid"
else
    fail "wrapper-skip-present-qwen3_5-hybrid" "expected exit 0 + hybrid deferred skip (rc=$rc out=$out)"
fi

# --- wrapper: per-family override beats MLXD_PARITY_CKPT_ROOT (N3) ---

WBOTH_DIR=$(mktemp -d)
mkdir -p "$WBOTH_DIR/override"
mkdir -p "$WBOTH_DIR/root/mlx-community/Qwen3-0.6B-4bit"
mkdir -p "$WBOTH_DIR/scripts"
cp "$WRAPPER" "$WBOTH_DIR/scripts/parity_family.sh"
cat > "$WBOTH_DIR/scripts/parity_temp0.sh" <<'STUBEOF'
#!/bin/sh
printf '%s\n' "$@" > "$(dirname "$0")/../delegate_args.txt"
exit 0
STUBEOF
chmod +x "$WBOTH_DIR/scripts/parity_temp0.sh"

MLXD_PARITY_CKPT_ROOT="$WBOTH_DIR/root" MLXD_PARITY_CKPT_QWEN3="$WBOTH_DIR/override" \
    sh "$WBOTH_DIR/scripts/parity_family.sh" qwen3 "test" >/dev/null 2>&1 && rc=0 || rc=$?
if [ "$rc" -eq 0 ] && [ -f "$WBOTH_DIR/delegate_args.txt" ]; then
    arg1=$(sed -n '1p' "$WBOTH_DIR/delegate_args.txt")
    if [ "$arg1" = "$WBOTH_DIR/override" ]; then
        pass "wrapper-override-beats-root"
    else
        fail "wrapper-override-beats-root" "wrong dir: $arg1 (expected override)"
    fi
else
    fail "wrapper-override-beats-root" "delegate not called (rc=$rc)"
fi
rm -rf "$WBOTH_DIR"

# --- wrapper: run-all with one failing family -> exit 1 (N4) ---

WFAIL_DIR=$(mktemp -d)
mkdir -p "$WFAIL_DIR/ckpt"
mkdir -p "$WFAIL_DIR/scripts"
cp "$WRAPPER" "$WFAIL_DIR/scripts/parity_family.sh"
cat > "$WFAIL_DIR/scripts/parity_temp0.sh" <<'STUBEOF'
#!/bin/sh
exit 1
STUBEOF
chmod +x "$WFAIL_DIR/scripts/parity_temp0.sh"

out=$(env -u MLXD_PARITY_CKPT_ROOT -u MLXD_PARITY_CKPT_GEMMA3 -u MLXD_PARITY_CKPT_QWEN2 \
    -u MLXD_PARITY_CKPT_LLAMA -u MLXD_PARITY_CKPT_MISTRAL -u MLXD_PARITY_CKPT_GEMMA4 \
    -u MLXD_PARITY_CKPT_QWEN3_5 \
    MLXD_PARITY_CKPT_QWEN3="$WFAIL_DIR/ckpt" \
    sh "$WFAIL_DIR/scripts/parity_family.sh" all 2>&1) && rc=0 || rc=$?
rm -rf "$WFAIL_DIR"
if [ "$rc" -eq 1 ]; then
    pass "wrapper-run-all-one-fail"
else
    fail "wrapper-run-all-one-fail" "expected exit 1 (rc=$rc)"
fi

exit "$FAILS"
