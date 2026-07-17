# mlxd

Modular C11 rewrite of the mlx-serve text-generation core. Unix-philosophy modules, OpenAI-compatible API surface only.

## Build

```bash
make              # debug-friendly build (no optimization)
make release      # -O2 -DNDEBUG
make debug        # -g -O0 -fsanitize=address,undefined
make tsan         # -g -O1 -fsanitize=thread (data race detection)
make test         # all CPU-only tests
make analyze      # Clang Static Analyzer via scan-build
make coverage     # clang source-based coverage -> build/cov/html/index.html
make clean
make install      # -> /usr/local/bin/mlxd
```

Requires: macOS (Apple Silicon), Homebrew mlx-c >= 0.6.0 + mlx >= 0.31.2 + libuv + llhttp + libcurl, clang with C11 support.

## Module graph

```
core (leaf)        - types, OpenAI DTOs, logging, sysinfo
mlxbridge (leaf)   - mlx-c FFI surface; links mlx-c + Metal frameworks
chat               - deps: core; links vendor/jinja_cpp
model              - deps: core, mlxbridge
engine             - deps: core, mlxbridge, model
registry           - deps: core
http               - deps: core, chat, model, engine, registry; links libuv + llhttp
cli (src/cli + main.c) - deps: all
```

Dependency direction: arrows point upward. `core` and `mlxbridge` are leaves. `engine` does NOT include `chat`. Enforcement is by convention and code review (C has no module-level import control).

## Architecture invariants

- **Single engine thread**: one dedicated thread is the sole mlx-c caller including all frees. No mlx calls from the event loop or callbacks. The engine thread owns all GPU state.
- **Engine emits token IDs, not text**: detokenization, thinking-split, and tool-call buffering run as chat-provided stream transforms on the event loop side.
- **Chat module is pure**: no tokenizer, no architecture awareness, no mlx includes. Template inputs arrive as a plain-data struct populated by the caller.
- **JSON via yyjson only**: inbound parsing and outbound serialization use yyjson exclusively. No manual JSON string construction or escape helpers.
- **No function pointer dispatch for engines**: concrete engine struct. HTTP layer takes an engine pointer directly; tests can substitute a mock engine without linking mlx.

## Threading model

- Single libuv event loop on main thread handles all network I/O (accept, HTTP parse, SSE write).
- Dedicated engine thread receives commands via MPSC mailbox (pthread mutex + cond).
- Commands: generate, embed, load, unload, reclaim, stop.
- Per-request stream: mutex/cond chunk buffer with cancel support. `uv_async_send` notifies the event loop when tokens are ready.
- Chunks: tagged union - token(id, logprob), done(finish_reason), err(message).

## Shutdown ordering

1. SIGINT sets atomic flag
2. Event loop stops accepting, closes listener
3. Engine rejects new work, cancels in-flight (streams end with done/cancelled)
4. Drain in-flight responses (bounded timeout)
5. Post stop command, join engine thread, cleanup
6. Second SIGINT exits 130

## Key conventions

- C11 (`-std=c11 -Wall -Wextra -Wpedantic`).
- Vendored jinja_cpp built from C++ source (`-std=c++17`), no prebuilt `.a`.
- Vendored yyjson built from source.
- libuv and llhttp via brew/pkg-config.
- mlx-c externs in mlxbridge are hand-written; link-time only. GPU smoke tests validate signatures.
- Tests are standalone `test_*.c` files in `tests/` with `main()` + `assert()`. Makefile `test` target runs all.
- Test fixtures in `tests/fixtures/`, path injected via `-DMLXD_FIXTURES_DIR`.
- `tokenizer_encode` (fixed-buffer, snprintf-style) is for tests and simple tools only; the server path must use `tokenizer_encode_alloc`.

## Subcommands

```
mlxd serve   # start OpenAI-compatible HTTP server
mlxd run     # one-shot or interactive text generation
mlxd pull    # download model from HuggingFace
mlxd list    # list locally available models
```

## Supported model families

gemma3, gemma4, qwen2, qwen3, qwen3_5, llama, mistral, lfm2, nemotron_h, bert, deepseek_v4.

## Explicitly deferred (v2)

Vision input, DiffusionGemma, text-model LoRA, multi-model residency with refcounting, continuous batching, adaptive spec gate, WebSocket/Responses/Ollama/Anthropic dialects.
