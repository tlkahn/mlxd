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

exit "$FAILS"
