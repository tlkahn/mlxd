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

Generate polls shutdown/cancel at prefill-chunk and decode-step grain (plus seed forward); residual non-interruptible window is at most one chunk or one pipelined decode step.

## mlx-c API pitfalls

- **`mlx_fast_rope` freqs convention**: the `freqs` array parameter expects `base^(2i/d)` (the period denominators, increasing values) - NOT `1/base^(2i/d)` (the angular frequencies). When implementing custom RoPE scheduling (llama3, proportional, yarn, etc.), compute `freq = pow(base, 2*i/d)`, apply scheduling adjustments, and pass the result directly. Cross-validate new RoPE variants against mlx-lm's corresponding Python class output, and always test end-to-end with a real checkpoint (`mlxd run`) since unit tests with synthetic weights can be self-consistently wrong.

## Parity pitfalls

- **Instruct raw completion loops vs oracle early-stop**: chat-tuned checkpoints (e.g. gemma4-it) often collapse under raw completion (no chat template) into short token cycles (`HelloHello...`, ` France is France is...`). That is expected without the template, not a forward-path bug. mlx-serve's serving-side `isDegenerateTailLoop` (period <= 8, 16 reps) stops early and reports `finish_reason=stop`; mlxd has no such guard, so temp-0 raw parity at `max_tokens=50` produces length-only mismatches on byte-identical prefixes. Prefer chat-mode parity for instruct models (`MLXD_PARITY_MODE=chat`: `/v1/chat/completions` vs `mlxd run --no-think`). Before blaming the forward path, confirm chat-templated output and logits parity against mlx-lm. Also check that `read_chat_template` actually loaded a template (`chat_template.jinja` fallback) - a missing template silently forces raw-like behavior.

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
mlxd run     # one-shot text generation
mlxd pull    # download model from HuggingFace
mlxd list    # list locally available models
```

## Supported model families

gemma3, gemma4, qwen2, qwen3, qwen3_5, llama, mistral, lfm2, nemotron_h, bert, deepseek_v3 (`MODEL_DEEPSEEK_V3`), deepseek_v3_2/deepseek_v32 (`MODEL_DEEPSEEK_V32`, MLA+MoE+DSA indexer), deepseek_v4 (`MODEL_DEEPSEEK_V4`). DeepSeek variants share MLA/MoE primitives but are separate `model_family_t` values - do not collapse V3/V3.2 into V4. HF string aliases `deepseek_v3_2` and `deepseek_v32` map together.

## Explicitly deferred (v2)

Vision input, DiffusionGemma, text-model LoRA, multi-model residency with refcounting, continuous batching, adaptive spec gate, WebSocket/Responses/Ollama/Anthropic dialects.
